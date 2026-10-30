import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

// Current stable API versions, verified against the vendors' documentation.
export const NOTION_VERSION = "2026-03-11";
const GITHUB_API_VERSION = "2026-03-10";
const COMMIT_LINK = "Commit Link";
const FULL_SHA = /^[0-9a-f]{40}$/i;

function required(env, name) {
  if (!env[name]?.trim()) {
    throw new Error(`Missing ${name}. Configure the repository secret or Actions environment.`);
  }
  return env[name].trim();
}

export function safeLog(message, env, output = console.log) {
  let text = String(message);
  for (const name of ["NOTION_TOKEN", "GITHUB_TOKEN", "NOTION_DATA_SOURCE_ID", "NOTION_OWNER_USER_ID"]) {
    const value = env[name]?.trim();
    if (value) text = text.replaceAll(value, "[REDACTED]");
  }
  // Keep untrusted API responses on one prefixed line, away from Actions commands.
  output(`[research-diary] ${text.replaceAll("\r", "\\r").replaceAll("\n", "\\n")}`);
}

async function requestJson(url, options, { fetchImpl, wait, log }) {
  for (let attempt = 0; ; attempt += 1) {
    let response;
    let text;
    try {
      response = await fetchImpl(url, {
        ...options,
        redirect: "error",
        signal: AbortSignal.timeout(30_000),
      });
      text = await response.text();
    } catch (error) {
      // A failed page-create request might already have succeeded server-side.
      // Never blindly retry an ambiguous write; a rerun checks Commit Link first.
      throw new Error(`${options.method} ${url} failed: ${error.message}`);
    }

    if (response.status === 429 && attempt < 3) {
      const seconds = Number(response.headers.get("retry-after") ?? 1);
      if (Number.isFinite(seconds) && seconds >= 0 && seconds <= 60) {
        log(`HTTP 429 from ${url}; retrying after ${seconds}s (${attempt + 1}/3).`);
        await wait(Math.max(1, seconds) * 1000);
        continue;
      }
    }
    if (!response.ok) {
      throw new Error(`${options.method} ${url}: HTTP ${response.status}; ${text}`);
    }
    try {
      return JSON.parse(text);
    } catch {
      throw new Error(`${options.method} ${url}: HTTP ${response.status}; invalid JSON response.`);
    }
  }
}

function normalizeCommit(sha, message, repositoryUrl) {
  if (typeof sha !== "string" || !FULL_SHA.test(sha)) {
    throw new Error("Commit metadata must contain a full 40-character commit SHA.");
  }
  if (typeof message !== "string") {
    throw new Error(`Commit ${sha} has no valid commit message.`);
  }
  return {
    sha: sha.toLowerCase(),
    message,
    url: `${repositoryUrl}/commit/${sha.toLowerCase()}`,
  };
}

async function githubRequest(path, env, http) {
  const apiUrl = (env.GITHUB_API_URL ?? "https://api.github.com").replace(/\/$/, "");
  return requestJson(`${apiUrl}/repos/${env.GITHUB_REPOSITORY}${path}`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${required(env, "GITHUB_TOKEN")}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": GITHUB_API_VERSION,
    },
  }, http);
}

async function selectCommits(event, env, repositoryUrl, http, log) {
  if (env.GITHUB_EVENT_NAME === "push") {
    const commits = event.commits ?? [];
    if (!Array.isArray(commits)) throw new Error("push.commits must be an array.");
    // Process the complete collection, including commits already seen on another branch.
    return commits.map(({ id, message }) => normalizeCommit(id, message, repositoryUrl));
  }

  const input = event.inputs?.commit_sha ?? "";
  if (typeof input !== "string") throw new Error("commit_sha must be a string.");
  const selectedSha = input.trim() || required(env, "GITHUB_SHA");
  if (!/^[0-9a-f]{7,40}$/i.test(selectedSha)) {
    throw new Error(`Cannot resolve manual commit_sha ${JSON.stringify(selectedSha)}: use a full SHA or an unambiguous 7–40 character hexadecimal abbreviation.`);
  }
  log(`Manual-test mode: resolving ${selectedSha}${input.trim() ? " (commit_sha input)" : ` (HEAD selected for ${env.GITHUB_REF ?? "this run"})`}.`);
  let commit;
  try {
    commit = await githubRequest(`/commits/${encodeURIComponent(selectedSha)}`, env, http);
  } catch (error) {
    throw new Error(`Cannot resolve manual commit ${selectedSha}: ${error.message}`);
  }
  if (typeof commit.sha !== "string" || !commit.sha.toLowerCase().startsWith(selectedSha.toLowerCase())) {
    throw new Error(`GitHub returned a different SHA for manual commit ${selectedSha}.`);
  }
  return [normalizeCommit(commit.sha, commit.commit?.message, repositoryUrl)];
}

function titleProperty(schema, label) {
  const titles = Object.entries(schema.properties ?? {}).filter(([, property]) => property.type === "title");
  if (titles.length !== 1) throw new Error(`${label} must have exactly one title property.`);
  return titles[0][0];
}

async function resolveProject(notion, projectDataSourceId, repositoryName, log) {
  const schema = await notion(`/data_sources/${encodeURIComponent(projectDataSourceId)}`);
  const property = titleProperty(schema, "Project data source");
  // Prefer the literal repository name, then its readable space-separated spelling.
  const candidates = new Set([repositoryName, repositoryName.replace(/[_-]+/g, " ")]);
  for (const name of candidates) {
    const result = await notion(`/data_sources/${encodeURIComponent(projectDataSourceId)}/query`, {
      filter: { property, title: { equals: name } },
      page_size: 2,
    });
    if (!Array.isArray(result.results)) throw new Error("Invalid Notion project-query response.");
    if (result.results.length > 1 || result.has_more) {
      throw new Error(`Multiple Notion projects match ${JSON.stringify(name)}; project names must be unique.`);
    }
    if (result.results.length === 1) {
      const project = result.results[0];
      const actualName = (project.properties?.[property]?.title ?? [])
        .map((part) => part.plain_text ?? part.text?.content ?? "").join("");
      if (!project.id || actualName.toLowerCase() !== name.toLowerCase()) {
        throw new Error("Notion returned a project without a matching name and page ID.");
      }
      log(`Resolved project: ${JSON.stringify(actualName)}.`);
      return project.id;
    }
  }
  throw new Error(`No Notion project matches repository ${JSON.stringify(repositoryName)}. Share the Project data source with the integration and check its project titles.`);
}

async function resolveOwner(notion, env, log) {
  const configuredId = env.NOTION_OWNER_USER_ID?.trim();
  if (configuredId) {
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(configuredId)) {
      throw new Error("NOTION_OWNER_USER_ID must be a native Notion user UUID.");
    }
    const user = await notion(`/users/${encodeURIComponent(configuredId)}`);
    if (user.type !== "person" || user.id !== configuredId) {
      throw new Error("NOTION_OWNER_USER_ID must resolve to a person, not an integration bot.");
    }
    log(`Resolved owner: ${JSON.stringify(user.name)} (native Notion person).`);
    return user.id;
  }

  const matches = [];
  const cursors = new Set();
  let cursor;
  do {
    const query = new URLSearchParams({ page_size: "100" });
    if (cursor) query.set("start_cursor", cursor);
    const result = await notion(`/users?${query}`);
    if (!Array.isArray(result.results)) throw new Error("Invalid Notion users response.");
    for (const user of result.results) {
      const name = (user.name ?? "").replaceAll(".", "").replace(/\s+/g, " ").trim().toLowerCase();
      if (user.type === "person" && ["julio m morales", "julio morales"].includes(name)) matches.push(user);
    }
    if (!result.has_more) break;
    cursor = result.next_cursor;
    if (!cursor || cursors.has(cursor)) throw new Error("Invalid Notion user pagination cursor.");
    cursors.add(cursor);
  } while (true);
  if (matches.length !== 1 || !matches[0].id) {
    throw new Error("Expected one Notion person named Julio M. Morales or Julio Morales. Configure NOTION_OWNER_USER_ID to select the verified account explicitly.");
  }
  log(`Resolved owner: ${JSON.stringify(matches[0].name)} (native Notion person).`);
  return matches[0].id;
}

// Shared by push and workflow_dispatch. Commit Link is the sole idempotency key.
export async function processCommit({ sha, message, url }, { notion, dataSourceId, titleProperty, getPageProperties, log, seen }) {
  log(`Processing commit ${sha}: ${url}`);
  if (seen.has(url)) {
    log(`Skipped existing commit ${sha} (already processed in this run).`);
    return "skipped";
  }
  const existing = await notion(`/data_sources/${encodeURIComponent(dataSourceId)}/query`, {
    filter: { property: COMMIT_LINK, url: { equals: url } },
    page_size: 1,
  });
  if (!Array.isArray(existing.results)) throw new Error(`Invalid Notion duplicate-query response for ${sha}.`);
  if (existing.results.length > 0) {
    seen.add(url);
    log(`Skipped existing commit ${sha} (matching Commit Link).`);
    return "skipped";
  }

  const subject = message.split(/\r\n|\n|\r/, 1)[0];
  // 1000 code points fit within 2000 UTF-16 units, including emoji, without splitting them.
  const title = (subject.match(/[\s\S]{1,1000}/gu) ?? []).map((content) => ({
    type: "text",
    text: { content },
  }));
  const page = await notion("/pages", {
    parent: { type: "data_source_id", data_source_id: dataSourceId },
    properties: {
      [titleProperty]: { title },
      [COMMIT_LINK]: { url },
      ...await getPageProperties(),
    },
  });
  if (page.object !== "page" || !page.id) throw new Error(`Invalid Notion page-create response for ${sha}; rerun to check whether it was recorded.`);
  seen.add(url);
  log(`Created Research Diary page for commit ${sha}.`);
  return "created";
}

export async function run({ env = process.env, fetchImpl = globalThis.fetch, wait = sleep, output = console.log } = {}) {
  const log = (message) => safeLog(message, env, output);
  const eventName = required(env, "GITHUB_EVENT_NAME");
  log(`Execution source: ${eventName}${eventName === "workflow_dispatch" ? " (manual-test mode)" : ""}.`);
  if (!["push", "workflow_dispatch"].includes(eventName)) {
    throw new Error(`Unsupported event: ${eventName}.`);
  }
  const event = JSON.parse(await readFile(required(env, "GITHUB_EVENT_PATH"), "utf8"));
  if (eventName === "push" && event.deleted) {
    log("Ignored deletion push; no commits to record.");
    return { created: 0, skipped: 0 };
  }
  const repository = required(env, "GITHUB_REPOSITORY");
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)) {
    throw new Error("GITHUB_REPOSITORY must be owner/repository.");
  }
  const repositoryUrl = `${(env.GITHUB_SERVER_URL ?? "https://github.com").replace(/\/$/, "")}/${repository}`;
  const http = { fetchImpl, wait, log };
  const commits = await selectCommits(event, env, repositoryUrl, http, log);
  if (commits.length === 0) {
    log("Push contains zero commits; nothing to record.");
    return { created: 0, skipped: 0 };
  }

  const token = required(env, "NOTION_TOKEN");
  const dataSourceId = required(env, "NOTION_DATA_SOURCE_ID");
  const notion = async (path, body) => requestJson(
    `https://api.notion.com/v1${path}`,
    {
      method: body === undefined ? "GET" : "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    },
    http,
  );
  const schema = await notion(`/data_sources/${encodeURIComponent(dataSourceId)}`);
  const title = titleProperty(schema, "Research Diary");
  if (schema.properties[COMMIT_LINK]?.type !== "url") {
    throw new Error('Research Diary must contain a URL property named exactly "Commit Link".');
  }
  const project = schema.properties.Project;
  if (project?.type !== "relation" || !project.relation?.data_source_id) {
    throw new Error('Research Diary needs a "Project" relation to an accessible Project data source. Share that data source with the integration.');
  }
  const ownerProperty = schema.properties.Owner ? "Owner" : "Person";
  if (schema.properties[ownerProperty]?.type !== "people") {
    throw new Error('Research Diary needs a native people property named "Owner" or "Person".');
  }
  let pageProperties;
  const getPageProperties = () => pageProperties ??= (async () => {
    const projectId = await resolveProject(notion, project.relation.data_source_id, repository.split("/")[1], log);
    const ownerId = await resolveOwner(notion, env, log);
    return {
      Project: { relation: [{ id: projectId }] },
      [ownerProperty]: { people: [{ object: "user", id: ownerId }] },
    };
  })();
  const context = { notion, dataSourceId, titleProperty: title, getPageProperties, log, seen: new Set() };
  const counts = { created: 0, skipped: 0 };
  for (const commit of commits) {
    try {
      const result = await processCommit(commit, context);
      counts[result] += 1;
    } catch (error) {
      throw new Error(`Commit ${commit.sha} failed: ${error.message}`);
    }
  }
  log(`Finished: ${counts.created} created, ${counts.skipped} skipped.`);
  return counts;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  run().catch((error) => {
    safeLog(`ERROR: ${error.message}`, process.env, console.error);
    process.exitCode = 1;
  });
}
