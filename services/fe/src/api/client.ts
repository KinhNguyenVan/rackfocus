import type { NeighborsResponse, SearchRequest, SearchResponse, TemporalPrepareResponse, TemporalSearchRequest, TemporalSearchResponse, TranscriptSuggestResponse } from "./types";
import { mockNeighbors, mockSearch, mockTemporalSearch } from "./mock";

// Mock là OPT-IN: phải khai `VITE_USE_MOCK=true` mới bật.
//
// Trước đây điều kiện là `!== "false"`, tức KHÔNG khai gì thì mock BẬT. Hậu quả: FE gọi
// mock trong khi backend thật đã chạy, và không có dấu hiệu nào ngoài chữ
// "frontend_mock" nằm lẫn trong `warnings` — người dùng thấy kết quả trông hợp lý (đúng
// video, đúng ảnh) nên tin là thật. Chính lúc thi thì đó là kiểu sai tệ nhất.
//
// Tệ hơn: `ops/runpod/Dockerfile` chạy `npm run build` mà không truyền biến này, nên
// bundle nướng trong image cũng ở chế độ mock. FE deploy lên RunPod TRẢ TOÀN DỮ LIỆU GIẢ.
//
// Chiều mặc định phải là an toàn: thiếu cấu hình nghĩa là "gọi backend thật" (sai thì lộ
// ra ngay bằng lỗi mạng), chứ không phải "trả dữ liệu giả" (sai thì im lặng).
const USE_MOCK = (import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_USE_MOCK === "true";

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

// GET /api/transcript/suggest — gợi ý scene theo keyword trong lời thoại (as-you-type).
// Không có mock: khi BE chưa cấu hình transcript_db_path sẽ trả 503, caller hiện lỗi nhẹ.
export async function transcriptSuggest(
	query: string,
	limit = 10,
	signal?: AbortSignal,
): Promise<TranscriptSuggestResponse> {
	const params = new URLSearchParams({ q: query, limit: String(limit) });
	const response = await fetch(`/api/transcript/suggest?${params}`, { signal });

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Transcript suggest failed (${response.status})`);
	}

	return response.json() as Promise<TranscriptSuggestResponse>;
}
