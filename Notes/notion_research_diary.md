# GitHub commits → Notion Research Diary

The **Notion Research Diary** workflow in `.github/workflows/notion-research-diary.yml`
records every commit in a GitHub push event's `commits` collection, on **any branch**.
It creates one page per commit, using only the first line of its message as the
title and `https://github.com/<owner>/<repo>/commit/<full-SHA>` as **Commit Link**.
Each new page also receives its matching **Project** relation, Julio M. Morales's
native Notion person in **Person** (or **Owner**, if present). Notion automatically
populates the existing **Date** creation timestamp when the page is created.
It has no branch or path filters. Deletion pushes are skipped; empty pushes do
nothing. The unfiltered push trigger also accepts tag pushes when GitHub includes
commits in their payload.

## Setup

1. [Create an internal Notion integration](https://developers.notion.com/guides/get-started/create-a-notion-integration)
   in the workspace containing Research Diary. Enable **Read content** and
   **Insert content** capabilities, plus **Read user information without email
   addresses** for native person lookup, and keep its integration token private.
2. Open the original Research Diary database in Notion and grant the integration
   access through the database's **Connections** menu. A linked view must point
   to the original database/data source to which the integration has access.
   Also connect the integration to the **Project Database** referenced by the
   diary's **Project** relation; relation targets must be accessible.
3. Ensure the Research Diary data source contains a property named exactly
   **Commit Link**, of type **URL**. The script discovers the title-type property
   automatically, so its name does not matter. No Commit SHA property is needed.
   It also requires a **Project** relation, a native people property named
   **Person** (or **Owner**). The live diary's existing **Date** is an automatic,
   read-only created-time property that Notion populates; the script does not
   add or write any date properties.
4. Open the database's settings → **Manage data sources** → select the Research
   Diary source → **Copy data source ID**. Use this ID, not a database, page, or
   view ID. If necessary, retrieve the database using `GET /v1/databases/<database_id>`
   with the integration and select the appropriate ID from its `data_sources`
   array. See Notion's [data source ID instructions](https://developers.notion.com/guides/get-started/upgrade-guide-2025-09-03#step-1-discover-the-data-sources-for-your-databases).
5. In GitHub → repository **Settings → Secrets and variables → Actions → New
   repository secret**, add:

   | Secret | Value |
   | --- | --- |
   | `NOTION_TOKEN` | The Notion integration token |
   | `NOTION_DATA_SOURCE_ID` | The Research Diary data source ID |
   | `NOTION_OWNER_USER_ID` (optional) | Julio's verified native Notion user ID; selects an account explicitly if names are ambiguous |

   Without `NOTION_OWNER_USER_ID`, the script paginates Notion's users and requires
   exactly one person named **Julio M. Morales** or **Julio Morales**. The connected
   account was verified through Notion's native user system and is displayed as
   **Julio Morales**. It writes a `people` property containing that user's ID,
   rather than text or the integration's bot identity. Missing or ambiguous
   people fail the job. Obtain a user ID through Notion's native user lookup or
   [`GET /v1/users`](https://developers.notion.com/reference/get-users); keep
   private IDs in GitHub configuration rather than committing them.

6. Commit and push the workflow, both scripts under `.github/scripts/`, and this
   documentation. Put the workflow on the repository's default branch to enable
   the manual **Run workflow** UI. Include the workflow and scripts in every
   branch you want to automate; GitHub evaluates push workflows from the pushed
   ref, so branches predating these files need to merge or cherry-pick them.
   After that, pushes to `main`, `develop`, feature, experiment, newly created,
   and other branches all qualify equally. No external server or service is needed.

The script uses native Node.js 24 `fetch` without npm dependencies. It uses
[Notion API version `2026-03-11`](https://developers.notion.com/reference/versioning),
queries `/v1/data_sources/<id>/query`, and creates pages with a `data_source_id`
parent. The workflow grants only GitHub `contents: read`; its automatically
provided `GITHUB_TOKEN` retrieves manual commit metadata without a separate PAT
or write access.

## Project, owner, and existing date

The same workflow and scripts work in any repository; there is no repository
allowlist or project page ID in the code. Each repository needs its own copy of
the workflow and scripts, and its own Actions secrets. GitHub does not apply a
workflow installed in one personal repository to other repositories, and it does
not share that repository's secrets with them.

For each repository whose name matches a project, install these four files on
its default branch and on existing branches that should record pushes:

- `.github/workflows/notion-research-diary.yml`
- `.github/scripts/notion-research-diary.mjs`
- `.github/scripts/notion-research-diary.test.mjs`
- `Notes/notion_research_diary.md`

Configure `NOTION_TOKEN` and `NOTION_DATA_SOURCE_ID` in that repository, and
optionally `NOTION_OWNER_USER_ID`. The same integration, Research Diary data
source, and native owner can be used across repositories. The integration needs
access to Research Diary and its related Project Database; individual project
page IDs are resolved at runtime. Verify each installation with **Run workflow**.
New branches inherit automation when created from a branch containing these
files. A newly created repository still needs this installation and its secrets;
creating a matching Notion project alone does not install a GitHub workflow.

The script reads the target data source from the diary's **Project** relation,
discovers that source's title property, and queries for the repository's name
(without the owner prefix). It prefers an exact name. If there is no exact
match, it tries the readable spelling with underscores/hyphens replaced by
spaces: **Morales_2025a** matches the existing **Morales 2025a** project. It fails
if no project matches or more than one matches; it never creates a replacement
project or guesses from unrelated project contents.

All commits in a run share the same resolved project and person. These lookups
happen only when a commit needs a new page and are cached for the remaining new
commits. Notion sets the existing **Date** to the page's creation time. A delayed
run or recovery run creates remaining pages with their actual creation time.
Existing matching commits are skipped without changing their project, person,
date, or title.

## Manual test from GitHub Actions

1. Open **GitHub → Actions → Notion Research Diary → Run workflow**.
2. Choose the branch/ref to test; it must contain the workflow and scripts.
3. Optionally enter **commit_sha**:
   - Supply a full SHA (recommended), or an unambiguous hexadecimal abbreviation
     of at least seven characters, to process that exact commit in this repository.
   - Leave it empty to process the HEAD commit of the selected branch/ref,
     captured as `GITHUB_SHA` when the run is dispatched.
4. Click **Run workflow** and inspect the **Record pushed or manually selected
   commits** step. It logs **manual-test mode**, resolves the SHA and actual
   message through GitHub's API, and processes exactly one commit.

Manual tests really create a diary page if the commit is new. They use the same
`processCommit` function as push runs. Repeated manual tests are safe: an existing
matching **Commit Link** is logged and skipped. An invalid or unresolvable SHA
fails the job with an explanation; no commit message input is needed.

## Duplicate protection and recovery

Before creating each page, the script queries the data source with the exact
filter `{"property":"Commit Link","url":{"equals":"<commit URL>"}}`.
Any match skips creation, including records created by another branch push,
an earlier manual test, or a partially completed run. Three commits pushed
together produce three lookups and up to three new pages; the `head_commit`
field does not replace the complete commit collection.

All push and manual runs share one repository-wide concurrency group. It uses
[`queue: max`](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#concurrency)
and does not cancel running jobs, so lookups and writes from this workflow cannot
race across branches. GitHub permits up to 100 pending runs in that queue; beyond
that platform limit, canceled runs need to be rerun. Notion does not enforce URL
uniqueness, so other independent writers to this data source must avoid inserting
the same commit URL concurrently.

Concurrency groups are scoped to a GitHub repository. Different repositories can
write to the shared diary independently because their canonical Commit Link URLs
include their distinct owner/repository names. Even the same Git SHA in two
repositories has two different commit URLs and belongs to each repository's own
matching project. Branches and repeated runs within one repository still share
the same URL and are skipped normally.

Unexpected HTTP errors, schema errors, and request timeouts fail the job. Logs
include the commit SHA, HTTP status, and API error details while redacting secrets.
HTTP 429 responses receive bounded retries honoring `Retry-After`; ambiguous
page-creation failures are never blindly retried. If a later commit fails after
earlier pages were created, fix the cause and use **Re-run failed jobs**: recorded
commits are skipped and remaining commits are processed. Keep Commit Link values
intact; deleting or changing them removes the duplicate key.

GitHub's own event limits still apply (for example, payload commit-count limits,
pushes affecting many refs, and commit messages that request skipping Actions).
The workflow processes the supplied collection and does not backfill history
outside it. A manual SHA run can record an omitted commit.

## Local verification

From the repository root, run:

```sh
node --check .github/scripts/notion-research-diary.mjs
node --test .github/scripts/notion-research-diary.test.mjs
```

The tests use temporary event payloads and mocked GitHub/Notion HTTP responses;
they never contact either service or require credentials. They exercise multiple
commits, title extraction, exact URL payloads and filtering, cross-branch and
manual duplicates, manual HEAD/SHA resolution, partial-failure recovery, deletion
and empty events, API failures, schema validation, and redacted logging.
They also test project name matching and ambiguity, native user pagination and
selection, leaving the automatic Date property to Notion, and multiple
repositories resolving different projects and commit URLs in the shared diary.
A real end-to-end test requires the two repository secrets, an accessible Notion data
source, Project Database access, user-information capability, and a GitHub Actions run.
