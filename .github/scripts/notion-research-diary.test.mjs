import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { NOTION_VERSION, run, safeLog } from "./notion-research-diary.mjs";

const SHA_A = "a".repeat(40);
const SHA_B = "b".repeat(40);
const SHA_C = "c".repeat(40);
const REPOSITORY = "example/research";
const DATA_SOURCE_ID = "example-data-source";
const PROJECT_DATA_SOURCE_ID = "example-projects";
const PROJECT_ID = "11111111-1111-4111-8111-111111111111";
const OWNER_ID = "22222222-2222-4222-8222-222222222222";
const commitUrl = (sha) => `https://github.com/${REPOSITORY}/commit/${sha}`;
const project = (name, id = PROJECT_ID) => ({ id, properties: { Project: { title: [{ plain_text: name }] } } });
const push = (...commits) => ({ deleted: false, commits });
const commit = (id, message = "Commit subject\n\nCommit body") => ({ id, message });
const response = (body, status = 200, headers = {}) => new Response(JSON.stringify(body), { status, headers });

async function fixture(t, event, eventName = "push") {
  const directory = await mkdtemp(join(tmpdir(), "notion-research-diary-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const eventPath = join(directory, "event.json");
  const env = {
    GITHUB_EVENT_NAME: eventName,
    GITHUB_EVENT_PATH: eventPath,
    GITHUB_REPOSITORY: REPOSITORY,
    GITHUB_SHA: SHA_B,
    GITHUB_REF: "refs/heads/feature/foo",
    GITHUB_TOKEN: "test-github-token",
    NOTION_TOKEN: "test-notion-token",
    NOTION_DATA_SOURCE_ID: DATA_SOURCE_ID,
  };
  const logs = [];
  const waits = [];
  const requests = [];
  const pages = new Map();
  const state = {
    schema: { properties: {
      "Entry name": { type: "title" }, "Commit Link": { type: "url" },
      Project: { type: "relation", relation: { data_source_id: PROJECT_DATA_SOURCE_ID } },
      Person: { type: "people" }, Date: { type: "created_time" },
    } },
    projectSchema: { properties: { Project: { type: "title" } } },
    projects: [project("research")],
    projectQueryOverride: null,
    users: [{ object: "user", type: "person", id: OWNER_ID, name: "Julio Morales" }],
    userStatus: 200,
    userPages: null,
    githubCommit: { sha: SHA_A, commit: { message: "Manual subject\n\nActual body from GitHub" } },
    githubStatus: 200,
    failCreateUrl: null,
    ambiguousCreate: false,
    rateLimit: false,
    queryOverride: null,
  };
  const fetchImpl = async (url, options) => {
    const body = options.body ? JSON.parse(options.body) : undefined;
    requests.push({ url, ...options, body });
    assert.equal(options.redirect, "error");
    assert.ok(options.signal instanceof AbortSignal);
    if (url.startsWith("https://api.github.com/")) {
      assert.equal(options.method, "GET");
      assert.equal(options.headers.Authorization, `Bearer ${env.GITHUB_TOKEN}`);
      assert.equal(options.headers["X-GitHub-Api-Version"], "2026-03-10");
      assert.ok(url.includes("/commits/"));
      return response(state.githubStatus === 200 ? state.githubCommit : { message: "No commit found for SHA" }, state.githubStatus);
    }
    assert.ok(url.startsWith("https://api.notion.com/v1/"), `Unexpected request: ${url}`);
    assert.equal(options.headers.Authorization, `Bearer ${env.NOTION_TOKEN}`);
    assert.equal(options.headers["Notion-Version"], NOTION_VERSION);
    const path = new URL(url).pathname;
    if (path === "/v1/users" || path.startsWith("/v1/users/")) {
      assert.equal(options.method, "GET");
      if (state.userStatus !== 200) return response({ code: "restricted_resource", message: "Enable user information capabilities" }, state.userStatus);
      if (path.startsWith("/v1/users/")) return response(state.users.find((user) => user.id === path.split("/").at(-1)) ?? {});
      if (state.userPages) return response(state.userPages(new URL(url).searchParams.get("start_cursor")));
      return response({ results: state.users, has_more: false });
    }
    if (path === `/v1/data_sources/${PROJECT_DATA_SOURCE_ID}`) return response(state.projectSchema);
    if (path === `/v1/data_sources/${PROJECT_DATA_SOURCE_ID}/query`) {
      assert.equal(options.method, "POST");
      assert.equal(body.filter.property, "Project");
      assert.equal(body.page_size, 2);
      if (state.projectQueryOverride) return state.projectQueryOverride(body);
      return response({ results: state.projects.filter((page) => page.properties.Project.title[0].plain_text.toLowerCase() === body.filter.title.equals.toLowerCase()), has_more: false });
    }
    if (path === `/v1/data_sources/${DATA_SOURCE_ID}`) {
      assert.equal(options.method, "GET");
      return response(state.schema);
    }
    if (path === `/v1/data_sources/${DATA_SOURCE_ID}/query`) {
      assert.equal(options.method, "POST");
      assert.equal(body.filter.property, "Commit Link");
      assert.equal(body.page_size, 1);
      if (state.rateLimit) {
        state.rateLimit = false;
        return response({ code: "rate_limited", message: "Slow down" }, 429, { "Retry-After": "2" });
      }
      if (state.queryOverride) return state.queryOverride();
      const page = pages.get(body.filter.url.equals);
      return response({ results: page ? [page] : [] });
    }
    assert.equal(path, "/v1/pages");
    assert.equal(options.method, "POST");
    const urlKey = body.properties["Commit Link"].url;
    if (state.failCreateUrl === urlKey) {
      return response({ code: "service_unavailable", message: "Try again later", request_id: "example-request" }, 503);
    }
    assert.ok(!pages.has(urlKey), `Attempted duplicate creation: ${urlKey}`);
    const page = { object: "page", id: `page-${pages.size + 1}`, ...body };
    pages.set(urlKey, page);
    if (state.ambiguousCreate) {
      state.ambiguousCreate = false;
      throw new Error("Connection lost after page was accepted");
    }
    return response(page);
  };
  const setEvent = async (nextEvent, nextName = env.GITHUB_EVENT_NAME) => {
    env.GITHUB_EVENT_NAME = nextName;
    await writeFile(eventPath, JSON.stringify(nextEvent));
  };
  await setEvent(event, eventName);
  return {
    env, logs, waits, requests, pages, state, setEvent,
    execute: () => run({ env, fetchImpl, wait: async (ms) => waits.push(ms), output: (line) => logs.push(line) }),
  };
}

test("push processes every commit, using only its subject and canonical permanent URL", async (t) => {
  const f = await fixture(t, {
    ...push(
      { ...commit(SHA_A, "Subject A\n\nBody A"), url: "https://untrusted.example/wrong" },
      commit(SHA_B, "Subject B\r\nBody B"),
      commit(SHA_C, 'Subject C $(echo unsafe) `echo unsafe` "quoted"\rBody C'),
    ),
    head_commit: commit(SHA_C, "Do not use head_commit"),
  });
  assert.deepEqual(await f.execute(), { created: 3, skipped: 0 });
  const titles = [...f.pages.values()].map((page) => page.properties["Entry name"].title.map((text) => text.text.content).join(""));
  assert.deepEqual(titles, ["Subject A", "Subject B", 'Subject C $(echo unsafe) `echo unsafe` "quoted"']);
  for (const sha of [SHA_A, SHA_B, SHA_C]) {
    const page = f.pages.get(commitUrl(sha));
    assert.deepEqual(page.parent, { type: "data_source_id", data_source_id: DATA_SOURCE_ID });
    assert.deepEqual(page.properties["Commit Link"], { url: commitUrl(sha) });
    assert.deepEqual(Object.keys(page.properties), ["Entry name", "Commit Link", "Project", "Person"]);
    assert.deepEqual(page.properties.Project, { relation: [{ id: PROJECT_ID }] });
    assert.deepEqual(page.properties.Person, { people: [{ object: "user", id: OWNER_ID }] });
    assert.equal(page.properties.Date, undefined);
  }
  const queries = f.requests.filter((request) => request.url.endsWith(`/data_sources/${DATA_SOURCE_ID}/query`));
  assert.deepEqual(queries.map((request) => request.body.filter), [SHA_A, SHA_B, SHA_C].map((sha) => ({ property: "Commit Link", url: { equals: commitUrl(sha) } })));
  assert.equal(f.requests.filter((request) => request.url.includes("/commits/")).length, 0);
  assert.equal(f.requests.filter((request) => request.url.startsWith("https://api.github.com")).length, 0);
  assert.ok(f.logs.some((line) => line.includes("Execution source: push")));
  assert.equal(f.logs.filter((line) => line.includes("Created Research Diary")).length, 3);
});

test("the same commit on main, develop, feature, experiment, or a new branch is skipped", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  assert.deepEqual(await f.execute(), { created: 1, skipped: 0 });
  for (const branch of ["main", "develop", "feature/foo", "experiment/bar", "new-branch"]) {
    f.env.GITHUB_REF = `refs/heads/${branch}`;
    await f.setEvent({ ...push(commit(SHA_A)), created: branch === "new-branch" });
    assert.deepEqual(await f.execute(), { created: 0, skipped: 1 });
  }
  assert.equal(f.pages.size, 1);
  assert.equal(f.logs.filter((line) => line.includes("Skipped existing commit")).length, 5);
});

test("a repeated SHA in one payload is also safe without relying on query visibility", async (t) => {
  const f = await fixture(t, push(commit(SHA_A), commit(SHA_A)));
  assert.deepEqual(await f.execute(), { created: 1, skipped: 1 });
  assert.equal(f.requests.filter((request) => request.url.endsWith(`/data_sources/${DATA_SOURCE_ID}/query`)).length, 1);
});

test("manual commit_sha resolves an abbreviation to one full SHA and uses GitHub's message", async (t) => {
  const f = await fixture(t, { inputs: { commit_sha: SHA_A.slice(0, 10).toUpperCase() }, commits: [commit(SHA_C)] }, "workflow_dispatch");
  assert.deepEqual(await f.execute(), { created: 1, skipped: 0 });
  assert.ok(f.requests[0].url.endsWith(`/commits/${SHA_A.slice(0, 10).toUpperCase()}`));
  assert.equal(f.pages.get(commitUrl(SHA_A)).properties["Entry name"].title[0].text.content, "Manual subject");
  assert.ok(f.logs.some((line) => line.includes("manual-test mode")));
  assert.ok(f.logs.some((line) => line.includes("commit_sha input")));
  assert.deepEqual(await f.execute(), { created: 0, skipped: 1 });
  assert.equal(f.pages.size, 1);
});

test("empty manual input resolves the dispatched HEAD snapshot and repeated runs skip it", async (t) => {
  const f = await fixture(t, { inputs: { commit_sha: "" } }, "workflow_dispatch");
  f.state.githubCommit = { sha: SHA_B, commit: { message: "Selected branch HEAD\nBody" } };
  assert.deepEqual(await f.execute(), { created: 1, skipped: 0 });
  assert.ok(f.requests[0].url.endsWith(`/commits/${SHA_B}`));
  assert.ok(f.logs.some((line) => line.includes("HEAD selected for refs/heads/feature/foo")));
  assert.deepEqual(await f.execute(), { created: 0, skipped: 1 });
  assert.equal(f.pages.size, 1);
});

test("manual execution skips an entry previously created by a push", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  await f.execute();
  await f.setEvent({ inputs: { commit_sha: SHA_A } }, "workflow_dispatch");
  assert.deepEqual(await f.execute(), { created: 0, skipped: 1 });
  assert.equal(f.pages.size, 1);
});

test("a nonexistent manual SHA fails clearly before any Notion call", async (t) => {
  const f = await fixture(t, { inputs: { commit_sha: SHA_A } }, "workflow_dispatch");
  f.state.githubStatus = 404;
  await assert.rejects(f.execute(), /Cannot resolve manual commit.*HTTP 404.*No commit found/);
  assert.equal(f.requests.length, 1);
  assert.equal(f.pages.size, 0);
});

test("malformed manual SHA is rejected without executing input or calling APIs", async (t) => {
  const f = await fixture(t, { inputs: { commit_sha: "$(echo unsafe)" } }, "workflow_dispatch");
  await assert.rejects(f.execute(), /Cannot resolve manual commit_sha.*hexadecimal abbreviation/);
  assert.equal(f.requests.length, 0);
});

test("GitHub must return the requested manual commit", async (t) => {
  const f = await fixture(t, { inputs: { commit_sha: SHA_B } }, "workflow_dispatch");
  await assert.rejects(f.execute(), /GitHub returned a different SHA/);
  assert.equal(f.requests.length, 1);
});

test("deletion pushes ignore even attached commits and need no credentials", async (t) => {
  const f = await fixture(t, { ...push(commit(SHA_A)), deleted: true });
  delete f.env.NOTION_TOKEN;
  delete f.env.NOTION_DATA_SOURCE_ID;
  delete f.env.GITHUB_REPOSITORY;
  assert.deepEqual(await f.execute(), { created: 0, skipped: 0 });
  assert.equal(f.requests.length, 0);
  assert.ok(f.logs.some((line) => line.includes("Ignored deletion push")));
});

test("empty or missing commit collections are graceful no-ops without credentials", async (t) => {
  const f = await fixture(t, push());
  delete f.env.NOTION_TOKEN;
  delete f.env.NOTION_DATA_SOURCE_ID;
  for (const event of [push(), {}]) {
    await f.setEvent(event);
    assert.deepEqual(await f.execute(), { created: 0, skipped: 0 });
  }
  assert.equal(f.requests.length, 0);
  assert.ok(f.logs.some((line) => line.includes("zero commits")));
});

test("partial failure exposes HTTP details and reruns recover without duplicating prior pages", async (t) => {
  const f = await fixture(t, push(commit(SHA_A), commit(SHA_B), commit(SHA_C)));
  f.state.failCreateUrl = commitUrl(SHA_B);
  await assert.rejects(f.execute(), /Commit b{40} failed:.*HTTP 503.*service_unavailable.*example-request/);
  assert.deepEqual([...f.pages.keys()], [commitUrl(SHA_A)]);
  f.state.failCreateUrl = null;
  assert.deepEqual(await f.execute(), { created: 2, skipped: 1 });
  assert.equal(f.pages.size, 3);
});

test("an ambiguous create failure is not retried and a later rerun finds the accepted page", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.ambiguousCreate = true;
  await assert.rejects(f.execute(), /Connection lost after page was accepted/);
  assert.equal(f.requests.filter((request) => request.url.endsWith("/pages")).length, 1);
  assert.deepEqual(await f.execute(), { created: 0, skipped: 1 });
  assert.equal(f.pages.size, 1);
});

test("duplicate lookup failure must fail instead of creating a page", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.queryOverride = () => response({ code: "restricted_resource", message: "Grant integration access" }, 403);
  await assert.rejects(f.execute(), /HTTP 403.*Grant integration access/);
  assert.equal(f.pages.size, 0);
});

test("rate limiting honors Retry-After and resumes the same duplicate check", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.rateLimit = true;
  assert.deepEqual(await f.execute(), { created: 1, skipped: 0 });
  assert.deepEqual(f.waits, [2000]);
  assert.ok(f.logs.some((line) => line.includes("HTTP 429")));
});

test("persistent rate limiting fails after bounded retries", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.queryOverride = () => response({ code: "rate_limited" }, 429, { "Retry-After": "1" });
  await assert.rejects(f.execute(), /HTTP 429.*rate_limited/);
  assert.deepEqual(f.waits, [1000, 1000, 1000]);
  assert.equal(f.pages.size, 0);
});

test("wrong Notion property types fail before any page writes", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.schema.properties["Commit Link"].type = "rich_text";
  await assert.rejects(f.execute(), /URL property named exactly "Commit Link"/);
  delete f.state.schema.properties["Commit Link"];
  await assert.rejects(f.execute(), /URL property named exactly "Commit Link"/);
  f.state.schema.properties = { "Commit Link": { type: "url" } };
  await assert.rejects(f.execute(), /exactly one title property/);
  assert.equal(f.pages.size, 0);
});

test("malformed Notion JSON and duplicate-query results fail safely", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.queryOverride = () => new Response("not-json");
  await assert.rejects(f.execute(), /HTTP 200; invalid JSON/);
  f.state.queryOverride = () => response({ results: null });
  await assert.rejects(f.execute(), /Invalid Notion duplicate-query response/);
  assert.equal(f.pages.size, 0);
});

test("long Unicode subjects are preserved in valid-size rich-text chunks", async (t) => {
  const subject = "🌞".repeat(2001);
  const f = await fixture(t, push(commit(SHA_A, `${subject}\nBody must not become title`)));
  await f.execute();
  const chunks = f.pages.get(commitUrl(SHA_A)).properties["Entry name"].title;
  assert.equal(chunks.map((part) => part.text.content).join(""), subject);
  assert.ok(chunks.every((part) => part.text.content.length <= 2000));
});

test("missing credentials, malformed push metadata, and unsupported events fail clearly", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  delete f.env.NOTION_TOKEN;
  await assert.rejects(f.execute(), /Missing NOTION_TOKEN/);
  await f.setEvent(push(commit("short-sha")));
  await assert.rejects(f.execute(), /full 40-character commit SHA/);
  await f.setEvent({ commits: "not-an-array" });
  await assert.rejects(f.execute(), /push.commits must be an array/);
  await f.setEvent({}, "pull_request");
  await assert.rejects(f.execute(), /Unsupported event: pull_request/);
  assert.equal(f.requests.length, 0);
});

test("log output redacts secrets and cannot introduce Actions command lines", () => {
  const output = [];
  const env = { NOTION_TOKEN: "token-value", GITHUB_TOKEN: "github-value", NOTION_DATA_SOURCE_ID: "source-value", NOTION_OWNER_USER_ID: "owner-value" };
  safeLog("HTTP 401 token-value github-value source-value owner-value\n::error::forged\rmessage", env, (line) => output.push(line));
  assert.equal(output[0], "[research-diary] HTTP 401 [REDACTED] [REDACTED] [REDACTED] [REDACTED]\\n::error::forged\\rmessage");
});

test("project lookup prefers the literal repository name and accepts its readable spelling", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.env.GITHUB_REPOSITORY = "example/Morales_2025a";
  f.state.projects = [project("Morales_2025a"), project("Morales 2025a", "another-project")];
  await f.execute();
  assert.equal([...f.pages.values()][0].properties.Project.relation[0].id, PROJECT_ID);
  assert.equal(f.requests.filter((request) => request.url.endsWith(`/data_sources/${PROJECT_DATA_SOURCE_ID}/query`)).length, 1);
  f.pages.clear();
  f.state.projects = [project("Morales 2025a")];
  await f.execute();
  assert.equal([...f.pages.values()][0].properties.Project.relation[0].id, PROJECT_ID);
  assert.ok(f.logs.some((line) => line.includes('Resolved project: "Morales 2025a"')));
});

test("missing or ambiguous project matches fail before creating pages", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.projects = [project("different project")];
  await assert.rejects(f.execute(), /No Notion project matches repository/);
  f.state.projects = [project("research"), project("research", "duplicate-project")];
  await assert.rejects(f.execute(), /Multiple Notion projects match/);
  assert.equal(f.pages.size, 0);
});

test("matching repositories use their own projects and commit URLs in the shared diary", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  const repositories = [
    ["Morales_2023", "Morales 2023"],
    ["Morales_2025a", "Morales 2025a"],
    ["Morales_202xa", "Morales 202xa"],
    ["Vesa_2026", "Vesa 2026"],
    ["another-repository", "another repository"],
  ];
  f.state.projects = repositories.map(([, name], index) => project(name, `project-${index}`));
  for (const [index, [repository]] of repositories.entries()) {
    f.env.GITHUB_REPOSITORY = `example/${repository}`;
    assert.deepEqual(await f.execute(), { created: 1, skipped: 0 });
    const url = `https://github.com/example/${repository}/commit/${SHA_A}`;
    assert.deepEqual(f.pages.get(url).properties.Project, { relation: [{ id: `project-${index}` }] });
    assert.equal(f.pages.get(url).properties["Commit Link"].url, url);
    assert.deepEqual(await f.execute(), { created: 0, skipped: 1 });
  }
  assert.equal(f.pages.size, repositories.length);
});

test("owner resolution uses native people and supports the full name or verified display name", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.users = [
    { type: "bot", id: "bot-id", name: "Julio M. Morales" },
    { type: "person", id: OWNER_ID, name: "Julio M. Morales" },
  ];
  await f.execute();
  assert.deepEqual(f.pages.get(commitUrl(SHA_A)).properties.Person, { people: [{ object: "user", id: OWNER_ID }] });
});

test("user lookup follows pagination and rejects missing or ambiguous people", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.userPages = (cursor) => cursor
    ? { results: f.state.users, has_more: false }
    : { results: [{ type: "person", name: "Someone else", id: "someone-else" }], has_more: true, next_cursor: "next-users" };
  await f.execute();
  assert.ok(f.requests.some((request) => request.url.includes("start_cursor=next-users")));
  f.pages.clear();
  f.state.userPages = null;
  f.state.users = [];
  await assert.rejects(f.execute(), /Expected one Notion person/);
  f.state.users = [
    { type: "person", id: OWNER_ID, name: "Julio Morales" },
    { type: "person", id: "another-user", name: "Julio M. Morales" },
  ];
  await assert.rejects(f.execute(), /Configure NOTION_OWNER_USER_ID/);
  assert.equal(f.pages.size, 0);
});

test("configured native user ID disambiguates accounts and rejects bot owners", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.env.NOTION_OWNER_USER_ID = OWNER_ID;
  f.state.users.push({ type: "person", id: "another-user", name: "Julio Morales" });
  await f.execute();
  assert.ok(f.requests.some((request) => request.url.endsWith(`/users/${OWNER_ID}`)));
  assert.equal(f.requests.filter((request) => request.url.includes("/users?")).length, 0);
  f.pages.clear();
  f.state.users[0].type = "bot";
  await assert.rejects(f.execute(), /person, not an integration bot/);
  f.env.NOTION_OWNER_USER_ID = "not-a-user-uuid";
  await assert.rejects(f.execute(), /native Notion user UUID/);
});

test("missing user information capability fails visibly without a page write", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.userStatus = 403;
  await assert.rejects(f.execute(), /HTTP 403.*Enable user information capabilities/);
  assert.equal(f.pages.size, 0);
});

test("automatic Date is left to Notion and a native Owner field is used when present", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.schema.properties.Owner = { type: "people" };
  await f.execute();
  const properties = f.pages.get(commitUrl(SHA_A)).properties;
  assert.equal(properties.Date, undefined);
  assert.deepEqual(properties.Owner, { people: [{ object: "user", id: OWNER_ID }] });
  assert.equal(properties.Person, undefined);
});

test("repeated manual execution skips existing pages without repeating enrichment lookups", async (t) => {
  const f = await fixture(t, { inputs: { commit_sha: SHA_A } }, "workflow_dispatch");
  await f.execute();
  f.state.users = [];
  f.state.projects = [];
  assert.deepEqual(await f.execute(), { created: 0, skipped: 1 });
  assert.equal(f.pages.size, 1);
  assert.equal(f.pages.get(commitUrl(SHA_A)).properties.Date, undefined);
});

test("project and people schemas are required before creating pages", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  f.state.schema.properties.Project.relation = {};
  await assert.rejects(f.execute(), /Project.*accessible Project data source/);
  f.state.schema.properties.Project.relation = { data_source_id: PROJECT_DATA_SOURCE_ID };
  f.state.schema.properties.Person.type = "rich_text";
  await assert.rejects(f.execute(), /native people property/);
  assert.equal(f.pages.size, 0);
});

test("the script entrypoint reports configuration failure and exits nonzero", async (t) => {
  const f = await fixture(t, push(commit(SHA_A)));
  const script = fileURLToPath(new URL("./notion-research-diary.mjs", import.meta.url));
  const result = spawnSync(process.execPath, [script], {
    env: { ...f.env, NOTION_TOKEN: "" },
    encoding: "utf8",
    timeout: 10_000,
  });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /ERROR: Missing NOTION_TOKEN/);
  assert.doesNotMatch(result.stdout + result.stderr, /test-github-token|test-notion-token/);
});
