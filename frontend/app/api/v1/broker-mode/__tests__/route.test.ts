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

const ORIGINAL_APEX_API_URL = process.env.APEX_API_URL;
const ORIGINAL_NEXT_PUBLIC_API_URL = process.env.NEXT_PUBLIC_API_URL;

function makeGetReq(token?: string) {
  return new NextRequest("http://localhost/api/v1/broker-mode", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });
}

function makePostReq(targetMode: string, token?: string) {
  return new NextRequest("http://localhost/api/v1/broker-mode", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ target_mode: targetMode }),
  });
}

describe("broker-mode route", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    process.env.APEX_API_URL = ORIGINAL_APEX_API_URL;
    process.env.NEXT_PUBLIC_API_URL = ORIGINAL_NEXT_PUBLIC_API_URL;
  });

  afterAll(() => {
    process.env.APEX_API_URL = ORIGINAL_APEX_API_URL;
    process.env.NEXT_PUBLIC_API_URL = ORIGINAL_NEXT_PUBLIC_API_URL;
  });

  it("returns broker mode on successful GET", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ broker_mode: "alpaca" }),
    } as Response);

    const res = await GET(makeGetReq("tok"));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.broker_mode).toBe("alpaca");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/ops/broker-mode/status",
      expect.objectContaining({
        cache: "no-store",
        headers: { authorization: "Bearer tok" },
      }),
    );
  });

  it("falls back to both when backend is unreachable", async () => {
    mockFetch.mockRejectedValueOnce(new Error("offline"));

    const res = await GET(makeGetReq());
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body).toEqual({ broker_mode: "both" });
  });

  it("forwards auth header and body on POST", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ status: "ok", broker_mode: "ibkr" }),
    } as Response);

    const res = await POST(makePostReq("ibkr", "secret"));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body).toEqual({ status: "ok", broker_mode: "ibkr" });
    expect(mockFetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/ops/broker-mode/change",
      expect.objectContaining({
        method: "POST",
        cache: "no-store",
        headers: {
          "Content-Type": "application/json",
          authorization: "Bearer secret",
        },
        body: JSON.stringify({ target_mode: "ibkr" }),
      }),
    );
  });

  it("uses NEXT_PUBLIC_API_URL when APEX_API_URL is unset", async () => {
    delete process.env.APEX_API_URL;
    process.env.NEXT_PUBLIC_API_URL = "http://frontend-proxy:9000/";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ broker_mode: "both" }),
    } as Response);

    await GET(makeGetReq());

    expect(mockFetch).toHaveBeenCalledWith(
      "http://frontend-proxy:9000/ops/broker-mode/status",
      expect.any(Object),
    );
  });
});
