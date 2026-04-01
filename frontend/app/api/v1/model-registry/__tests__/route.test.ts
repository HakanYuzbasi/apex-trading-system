/**
 * @jest-environment node
 */
import { jest } from "@jest/globals";
import { NextRequest } from "next/server";

const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

const makeReq = (token?: string) =>
  new NextRequest("http://localhost/api/v1/model-registry", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null, ok = true) {
  if (!ok || payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const FULL_SNAPSHOT = {
  available: true,
  total_models: 1,
  total_versions: 3,
  models: {
    god_level: {
      champion_id: "god_level_1700000100",
      champion_ic: 0.12,
      champion_sharpe: 1.4,
      champion_promoted_at: 1700000100,
      total_versions: 3,
      versions: [
        { version_id: "god_level_1700000100", model_name: "god_level", status: "champion", metrics: { ic: 0.12, sharpe: 1.4 } },
        { version_id: "god_level_1699913700", model_name: "god_level", status: "retired", metrics: { ic: 0.10, sharpe: 1.2 } },
      ],
    },
  },
  recent_events: [
    { ts: 1700000100, version_id: "god_level_1700000100", action: "promote", reason: "IC 0.1200 beats champion 0.1000" },
  ],
  ic_promote_delta: 0.005,
  ic_rollback_thresh: -0.01,
  sharpe_min: 0.2,
};

afterEach(() => jest.clearAllMocks());

describe("GET /api/v1/model-registry", () => {
  it("returns full snapshot when backend succeeds", async () => {
    mockBackend(FULL_SNAPSHOT);
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(body.total_models).toBe(1);
    expect(body.models.god_level.champion_ic).toBe(0.12);
  });

  it("forwards auth header to backend", async () => {
    mockBackend(FULL_SNAPSHOT);
    await GET(makeReq("mytoken"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/model-registry"),
      expect.objectContaining({ headers: { authorization: "Bearer mytoken" } })
    );
  });

  it("returns fallback on backend 503", async () => {
    mockBackend(null, false);
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.models).toEqual({});
    expect(body.recent_events).toEqual([]);
  });

  it("returns fallback on fetch exception", async () => {
    mockFetch.mockRejectedValueOnce(new Error("ECONNREFUSED"));
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.note).toMatch(/unreachable/i);
  });

  it("passes through recent_events", async () => {
    mockBackend(FULL_SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.recent_events).toHaveLength(1);
    expect(body.recent_events[0].action).toBe("promote");
  });

  it("passes through ic_promote_delta and sharpe_min", async () => {
    mockBackend(FULL_SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.ic_promote_delta).toBe(0.005);
    expect(body.sharpe_min).toBe(0.2);
  });

  it("passes through model versions list", async () => {
    mockBackend(FULL_SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.models.god_level.versions).toHaveLength(2);
    expect(body.models.god_level.versions[0].status).toBe("champion");
  });

  it("works without auth token", async () => {
    mockBackend({ available: true, models: {}, total_models: 0, recent_events: [] });
    const res = await GET(makeReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
  });
});
