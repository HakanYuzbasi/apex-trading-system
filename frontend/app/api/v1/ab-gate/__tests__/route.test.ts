/**
 * @jest-environment node
 */
import { jest } from "@jest/globals";
import { NextRequest } from "next/server";

const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET, POST } = require("../route") as {
  GET: (req: NextRequest) => Promise<Response>;
  POST: (req: NextRequest) => Promise<Response>;
};

const makeGetReq = (token?: string) =>
  new NextRequest("http://localhost/api/v1/ab-gate", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

const makePostReq = (body: object, token?: string) =>
  new NextRequest("http://localhost/api/v1/ab-gate", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
  });

function mockBackend(payload: object | null, status = 200) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({
      ok: status < 400,
      status,
      json: async () => payload,
    } as Response);
  }
}

const MOCK_STATUS = {
  available: true,
  control: {
    name: "control",
    weights: { ml: 0.5, tech: 0.3, sentiment: 0.2 },
    alpha: 35.0,
    beta_: 25.0,
    n_trades: 58,
    win_rate_mean: 0.583,
  },
  challenger: {
    name: "ml-heavy-v2",
    weights: { ml: 0.7, tech: 0.2, sentiment: 0.1 },
    alpha: 12.0,
    beta_: 8.0,
    n_trades: 18,
    win_rate_mean: 0.600,
  },
  p_challenger_better: 0.62,
  promotions: 1,
  promotion_history: [],
  thresholds: { min_trades: 30, promotion_prob: 0.95, rollback_drop: 0.05, shadow_hours: 48 },
};

describe("GET /api/v1/ab-gate", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with available=true", async () => {
    mockBackend(MOCK_STATUS);
    const res = await GET(makeGetReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
  });

  it("passes control variant through", async () => {
    mockBackend(MOCK_STATUS);
    const body = await (await GET(makeGetReq())).json();
    expect(body.control.name).toBe("control");
    expect(body.control.n_trades).toBe(58);
  });

  it("passes challenger variant through", async () => {
    mockBackend(MOCK_STATUS);
    const body = await (await GET(makeGetReq())).json();
    expect(body.challenger.name).toBe("ml-heavy-v2");
  });

  it("passes p_challenger_better through", async () => {
    mockBackend(MOCK_STATUS);
    const body = await (await GET(makeGetReq())).json();
    expect(body.p_challenger_better).toBeCloseTo(0.62);
  });

  it("returns available=false on backend failure", async () => {
    mockBackend(null);
    const body = await (await GET(makeGetReq())).json();
    expect(body.available).toBe(false);
  });

  it("handles fetch rejection gracefully", async () => {
    mockFetch.mockRejectedValueOnce(new Error("timeout"));
    const body = await (await GET(makeGetReq())).json();
    expect(body.available).toBe(false);
  });

  it("forwards auth header", async () => {
    mockBackend(MOCK_STATUS);
    await GET(makeGetReq("tok42"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/ab-gate"),
      expect.objectContaining({
        headers: expect.objectContaining({ authorization: "Bearer tok42" }),
      })
    );
  });
});

describe("POST /api/v1/ab-gate (register challenger)", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns registered=true on success", async () => {
    mockBackend({ registered: true, name: "v2", weights: { ml: 0.7, tech: 0.3 } });
    const body = await (
      await POST(makePostReq({ weights: { ml: 0.7, tech: 0.3 }, name: "v2" }))
    ).json();
    expect(body.registered).toBe(true);
  });

  it("forwards weights in POST body", async () => {
    mockBackend({ registered: true, name: "v2", weights: {} });
    await POST(makePostReq({ weights: { ml: 0.8 }, name: "v2" }, "tok"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/ab-gate/register"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("returns 500 on network failure", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network down"));
    const res = await POST(makePostReq({ weights: { ml: 0.5 } }));
    expect(res.status).toBe(500);
  });
});
