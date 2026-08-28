import { useEffect, useRef, useState } from "react";
import { searchTemporal } from "../api/client";
import type { TemporalChain } from "../api/types";

export function useTemporalSearch(
	event1: string,
	event2: string,
	useLlm = false,
	exact = false,
	tags: number[] | null = null,
) {
	const [chains, setChains] = useState<TemporalChain[]>([]);
	const [totalMs, setTotalMs] = useState<number | null>(null);
	const [warnings, setWarnings] = useState<string[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const requestId = useRef(0);

	useEffect(() => {
		const e1 = event1.trim();
		const e2 = event2.trim();
		if (!e1 || !e2) {
			setChains([]);
			setTotalMs(null);
			setWarnings([]);
			setError(null);
			setLoading(false);
			return;
		}

		const controller = new AbortController();
		const currentRequest = ++requestId.current;
		setLoading(true);
		setError(null);

		// `tags ?? undefined`: JSON.stringify bỏ hẳn khoá undefined, nên BE nhận `tags`
		// vắng mặt -> None -> giữ hành vi cũ theo use_llm. Gửi [] là chuyện khác hẳn:
		// "user đã bỏ tick hết, search toàn kho".
		searchTemporal(
			{ event1: e1, event2: e2, use_llm: useLlm, exact, tags: tags ?? undefined },
			controller.signal,
		)
			.then((data) => {
				if (currentRequest !== requestId.current) return;
				setChains(data.chains ?? []);
				setTotalMs(data.timings_ms?.total ?? null);
				setWarnings(data.warnings ?? []);
			})
			.catch((cause: unknown) => {
				if (controller.signal.aborted || currentRequest !== requestId.current) return;
				setError(cause instanceof Error ? cause.message : "Temporal search failed");
				setChains([]);
				setTotalMs(null);
				setWarnings([]);
			})
			.finally(() => {
				if (currentRequest === requestId.current) setLoading(false);
			});

		return () => controller.abort();
	// tags là mảng -> mỗi lần render là một identity mới, đưa thẳng vào deps sẽ chạy lại
	// vô hạn. Serialize thành chuỗi. null (không qua prepare) khác [] (bỏ tick hết) nên
	// hai cái phải cho ra hai khoá khác nhau.
	}, [event1, event2, useLlm, exact, tags === null ? "null" : tags.join(",")]);

	return { chains, totalMs, warnings, error, loading };
}
