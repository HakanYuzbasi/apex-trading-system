/**
 * @jest-environment node
 */
import { jest } from "@jest/globals";
import { NextRequest } from "next/server";

const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

const makeReq = (token?: string, n?: string) => {
  const url = n
    ? `http://localhost/api/v1/alerts?n=${n}`
    : "http://localhost/api/v1/alerts";
  return new NextRequest(url, {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });
};

function mockBackend(payload: object | null, ok = true) {
  if (!ok || payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

afterEach(() => jest.clearAllMocks());

describe("GET /api/v1/alerts", () => {
  it("returns available:true with alerts when backend succeeds", async () => {
    const alerts = [{ event_type: "KILL_SWITCH", message: "halted", ts: 1000, channel: "telegram" }];
    mockBackend({ available: true, channel: "telegram", alerts, total_buffered: 1 });
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(body.alerts).toHaveLength(1);
    expect(body.alerts[0].event_type).toBe("KILL_SWITCH");
  });

  it("forwards auth header to backend", async () => {
    mockBackend({ available: true, alerts: [] });
    await GET(makeReq("mytoken"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/alerts"),
      expect.objectContaining({ headers: { authorization: "Bearer mytoken" } })
    );
  });

  it("passes n query param to backend", async () => {
    mockBackend({ available: true, alerts: [] });
    await GET(makeReq(undefined, "20"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("n=20"),
      expect.anything()
    );
  });

  it("returns available:false on backend failure", async () => {
    mockBackend(null, false);
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.alerts).toEqual([]);
  });

  it("returns available:false on fetch error", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network error"));
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.note).toMatch(/unreachable/i);
  });

  it("passes through channel field", async () => {
    mockBackend({ available: true, channel: "telegram+slack", alerts: [] });
    const body = await (await GET(makeReq())).json();
    expect(body.channel).toBe("telegram+slack");
  });

  it("passes through total_buffered", async () => {
    mockBackend({ available: true, alerts: [], total_buffered: 42 });
    const body = await (await GET(makeReq())).json();
    expect(body.total_buffered).toBe(42);
  });

  it("works without auth token", async () => {
    mockBackend({ available: true, alerts: [{ event_type: "DRAWDOWN" }] });
    const res = await GET(makeReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.alerts[0].event_type).toBe("DRAWDOWN");
  });
});
