import type { NeighborsResponse, SearchRequest, SearchResponse, TemporalPrepareResponse, TemporalSearchRequest, TemporalSearchResponse } from "./types";
import { mockNeighbors, mockSearch, mockTemporalSearch } from "./mock";

const USE_MOCK = (import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_USE_MOCK !== "false";

export async function search(request: SearchRequest, signal?: AbortSignal): Promise<SearchResponse> {
	if (USE_MOCK) return Promise.resolve(mockSearch(request));

	const response = await fetch("/api/search", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(request),
		signal,
	});

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Search failed (${response.status})`);
	}

	return response.json() as Promise<SearchResponse>;
}

export async function neighbors(
	keyframeUrl: string,
	before = 25,
	after = 25,
	toKey?: string,
	signal?: AbortSignal,
): Promise<NeighborsResponse> {
	if (USE_MOCK) return Promise.resolve(mockNeighbors(keyframeUrl, before, after));

	const params = new URLSearchParams({
		key: keyframeUrl,
		before: String(before),
		after: String(after),
	});
	if (toKey) params.set("to_key", toKey);
	const response = await fetch(`/api/neighbors?${params}`, { signal });

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Neighbors failed (${response.status})`);
	}

	return response.json() as Promise<NeighborsResponse>;
}

export async function searchTemporal(
	request: TemporalSearchRequest,
	signal?: AbortSignal,
): Promise<TemporalSearchResponse> {
	if (USE_MOCK) return Promise.resolve(mockTemporalSearch(request));

	const response = await fetch("/api/search/temporal", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(request),
		signal,
	});

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Temporal search failed (${response.status})`);
	}

	return response.json() as Promise<TemporalSearchResponse>;
}

// `useLlm` chỉ bật/tắt bước chọn tag; tách đoạn luôn chạy (đã gọi prepare nghĩa là user
// đã bật tách event). false -> BE bỏ hẳn lời gọi enrich, trả tags rỗng.
export async function prepareTemporal(
	query: string,
	useLlm = true,
	signal?: AbortSignal,
): Promise<TemporalPrepareResponse> {
	const response = await fetch("/api/search/temporal/prepare", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ query, use_llm: useLlm }),
		signal,
	});

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Prepare failed (${response.status})`);
	}

	return response.json() as Promise<TemporalPrepareResponse>;
}
