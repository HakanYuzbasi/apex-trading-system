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
  new NextRequest("http://localhost/api/v1/hmm-regime", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null, ok = true) {
  if (!ok || payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

afterEach(() => jest.clearAllMocks());

describe("GET /api/v1/hmm-regime", () => {
  it("returns available:true with label when backend succeeds", async () => {
    mockBackend({ available: true, method: "hmm", current_label: "bull", confidence: 0.82 });
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(body.current_label).toBe("bull");
  });

  it("forwards auth header to backend", async () => {
    mockBackend({ available: true, current_label: "neutral" });
    await GET(makeReq("mytoken"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/hmm-regime"),
      expect.objectContaining({ headers: { authorization: "Bearer mytoken" } })
    );
  });

  it("returns available:false on backend failure", async () => {
    mockBackend(null, false);
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
  });

  it("returns available:false on fetch error", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network error"));
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.note).toMatch(/unreachable/i);
  });

  it("passes through state_probs", async () => {
    mockBackend({ available: true, state_probs: { bull: 0.82, neutral: 0.10, bear: 0.06, volatile: 0.02 } });
    const body = await (await GET(makeReq())).json();
    expect(body.state_probs.bull).toBeCloseTo(0.82);
  });

  it("passes through viterbi_path", async () => {
    mockBackend({ available: true, viterbi_path: ["bull", "bull", "neutral"] });
    const body = await (await GET(makeReq())).json();
    expect(body.viterbi_path).toEqual(["bull", "bull", "neutral"]);
  });

  it("passes through current_vix_regime", async () => {
    mockBackend({ available: true, current_vix_regime: "bear" });
    const body = await (await GET(makeReq())).json();
    expect(body.current_vix_regime).toBe("bear");
  });

  it("works without auth token", async () => {
    mockBackend({ available: true, current_label: "volatile" });
    const res = await GET(makeReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.current_label).toBe("volatile");
  });
});
