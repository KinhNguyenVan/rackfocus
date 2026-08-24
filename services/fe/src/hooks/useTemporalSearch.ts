import { useEffect, useRef, useState } from "react";
import { searchTemporal } from "../api/client";
import type { TemporalChain } from "../api/types";

export function useTemporalSearch(
	event1: string,
	event2: string,
	useLlm = false,
	exact = false,
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

		searchTemporal({ event1: e1, event2: e2, use_llm: useLlm, exact }, controller.signal)
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
	}, [event1, event2, useLlm, exact]);

	return { chains, totalMs, warnings, error, loading };
}
