from __future__ import annotations

import asyncio
import io
import json
import math
import re
import threading
import time
import traceback
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import quote

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from optimal_cpd_omega_prime import (
    SearchCancelled,
    assign_labels,
    dp_best,
    exhaustive_best_for_k,
)


app = FastAPI(title="CPD Calculator API")

# --- Exhaustive-search cost budget -----------------------------------------
# Measured on this codebase: 1,560,780 candidates in 36.8 s ≈ 42,000/s. The
# limits below are expressed in candidates so the UI can quote a cost before
# the user commits, instead of discovering it by waiting.
EXHAUSTIVE_RATE_PER_SEC = 42_000
EXHAUSTIVE_AUTO_LIMIT = 500_000          # ≈12 s — runs without asking
EXHAUSTIVE_CONFIRM_LIMIT = 20_000_000    # ≈8 min — needs an explicit opt-in
HEARTBEAT_SECONDS = 2.0                  # progress cadence during a long search

SCORE_COLUMN_NAMES = {
    "score",
    "scores",
    "performance",
    "value",
    "total",
    "คะแนน",
}

UPPER_BOUND, LOWER_BOUND = 100.0, 0.0

# Allow CORS for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Content-Disposition is not CORS-safelisted, so without this the browser
    # hides it and every export downloads under a generic fallback name.
    expose_headers=["Content-Disposition"],
)


class CalculationFlowError(Exception):
    """An error that can be shown safely in the execution log."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def bilingual(thai: str, english: str) -> str:
    """One message the Thai UI can show and an English log can still grep."""
    return f"{thai} · {english}"


async def read_table(contents: bytes, filename: str) -> tuple[pd.DataFrame, str]:
    """Parse an uploaded CSV/XLSX into a DataFrame, or explain why we cannot."""
    filename_lower = filename.lower()
    source = io.BytesIO(contents)
    try:
        if filename_lower.endswith(".csv"):
            return await asyncio.to_thread(pd.read_csv, source), "CSV"
        if filename_lower.endswith((".xls", ".xlsx")):
            return await asyncio.to_thread(pd.read_excel, source), "Excel"
    except Exception as exc:  # malformed file, wrong encoding, corrupt workbook
        raise CalculationFlowError(
            400,
            bilingual(
                f"เปิดไฟล์ “{filename}” ไม่สำเร็จ ไฟล์อาจเสียหายหรือไม่ใช่ตารางข้อมูล "
                f"({type(exc).__name__})",
                f"Could not parse {filename}: {exc}",
            ),
        ) from exc
    raise CalculationFlowError(
        400,
        bilingual(
            f"ไฟล์ “{filename}” ไม่ใช่ .csv หรือ .xlsx กรุณาแปลงไฟล์ก่อนอัปโหลด",
            "Unsupported file format. Please upload CSV or XLSX.",
        ),
    )


def select_score_column(df: pd.DataFrame) -> tuple[Any, str]:
    """Pick the score column by name, else fall back to the last numeric one."""
    named = next(
        (column for column in df.columns if str(column).strip().lower() in SCORE_COLUMN_NAMES),
        None,
    )
    if named is not None:
        return named, "ชื่อคอลัมน์มาตรฐาน"

    numeric_columns = df.select_dtypes(include=[np.number])
    if numeric_columns.empty:
        raise CalculationFlowError(
            400,
            bilingual(
                "ไม่พบคอลัมน์ตัวเลขในไฟล์ ต้องมีคอลัมน์คะแนนอย่างน้อยหนึ่งคอลัมน์ "
                "(ตั้งชื่อว่า score, total หรือ คะแนน จะแม่นยำที่สุด)",
                "No numeric columns found in data.",
            ),
        )
    # Documented behaviour is the LAST numeric column; the help text, the log,
    # and this line have to agree or the trace stops being auditable.
    return numeric_columns.columns[-1], "คอลัมน์ตัวเลขคอลัมน์สุดท้าย"


def extract_scores(df: pd.DataFrame, score_column: Any) -> tuple[np.ndarray, int]:
    """Return scores sorted descending, plus how many rows were unusable."""
    numeric_scores = pd.to_numeric(df[score_column], errors="coerce").to_numpy(dtype=float)
    finite_scores = numeric_scores[np.isfinite(numeric_scores)]
    removed_count = len(numeric_scores) - len(finite_scores)
    if len(finite_scores) == 0:
        raise CalculationFlowError(
            400,
            bilingual(
                f"คอลัมน์ “{score_column}” ไม่มีคะแนนที่เป็นตัวเลขเลย "
                "ตรวจสอบว่าเลือกคอลัมน์ถูกต้องและค่าไม่ได้ถูกบันทึกเป็นข้อความ",
                f"No valid numeric scores found in column {score_column}.",
            ),
        )
    return np.sort(finite_scores)[::-1], removed_count


def parse_labels(labels: str) -> list[str]:
    """Parse |L| from the comma-separated field, refusing anything ambiguous.

    |L| is a fixed input by mathematical necessity, so a label list that cannot
    be counted unambiguously is an error, not something to silently repair.
    """
    symbols = [symbol.strip() for symbol in labels.split(",") if symbol.strip()]
    if not symbols:
        raise CalculationFlowError(
            400,
            bilingual(
                "ยังไม่ได้ระบุระดับเกรด กรอกอย่างน้อย 2 ระดับ คั่นด้วยจุลภาค เช่น A, B, C",
                "Labels cannot be empty.",
            ),
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for symbol in symbols:
        if symbol in seen and symbol not in duplicates:
            duplicates.append(symbol)
        seen.add(symbol)
    if duplicates:
        raise CalculationFlowError(
            400,
            bilingual(
                f"ระดับเกรดซ้ำกัน: {', '.join(duplicates)} — |L| ต้องนับได้ชัดเจน "
                "ลบตัวที่ซ้ำออกก่อนคำนวณ",
                f"Duplicate grade labels: {', '.join(duplicates)}.",
            ),
        )

    if len(symbols) < 2:
        raise CalculationFlowError(
            400,
            bilingual(
                "ต้องมีระดับเกรดอย่างน้อย 2 ระดับ การแบ่งกลุ่มเดียวทำให้ Ω′ = 1 เสมอ "
                "ซึ่งไม่มีความหมายในการเปรียบเทียบ",
                "At least 2 grade labels are required.",
            ),
        )
    return symbols


def check_label_budget(sample_size: int, num_labels: int) -> None:
    """Refuse to run when n < |L| instead of quietly shrinking k.

    `min(num_labels, n)` would make the system sweep k on its own, which is
    exactly what Ω′ cannot survive: the metric only compares methods at a fixed
    label budget.
    """
    if sample_size < num_labels:
        raise CalculationFlowError(
            400,
            bilingual(
                f"ข้อมูลมี {sample_size:,} คะแนน แต่ระบุ |L| = {num_labels} ระดับ "
                f"ระบบจะไม่ลดจำนวนเกรดให้เอง เพราะ Ω′ เปรียบเทียบได้เฉพาะที่ |L| คงที่ "
                f"— ลดระดับเกรดเหลือไม่เกิน {sample_size} ระดับ หรือใช้ข้อมูลที่ใหญ่ขึ้น",
                f"n={sample_size} is smaller than |L|={num_labels}; "
                "k is a fixed input and will not be reduced automatically.",
            ),
        )


def exhaustive_budget(sample_size: int, cluster_count: int) -> dict[str, Any]:
    """Quote the cost of the exhaustive search before anyone commits to it."""
    candidate_count = math.comb(sample_size - 1, cluster_count - 1)
    estimated_seconds = candidate_count / EXHAUSTIVE_RATE_PER_SEC
    if candidate_count <= EXHAUSTIVE_AUTO_LIMIT:
        mode = "auto"
    elif candidate_count <= EXHAUSTIVE_CONFIRM_LIMIT:
        mode = "confirm"
    else:
        mode = "refused"
    return {
        "candidate_count": candidate_count,
        "candidate_formula": f"C({sample_size - 1},{cluster_count - 1})",
        "estimated_seconds": round(estimated_seconds, 1),
        "mode": mode,
        "auto_limit": EXHAUSTIVE_AUTO_LIMIT,
        "confirm_limit": EXHAUSTIVE_CONFIRM_LIMIT,
    }


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return "ไม่ถึง 1 วินาที"
    if seconds < 90:
        return f"~{seconds:.0f} วินาที"
    if seconds < 5400:
        return f"~{seconds / 60:.0f} นาที"
    return f"~{seconds / 3600:.1f} ชั่วโมง"


def data_warnings(
    sample_size: int,
    num_labels: int,
    removed_count: int,
    budget: dict[str, Any],
    force_exhaustive: bool,
) -> list[dict[str, str]]:
    """Say out loud everything that makes a result less trustworthy."""
    warnings: list[dict[str, str]] = []
    if num_labels < 3:
        warnings.append({
            "level": "warning",
            "text": (
                f"|L| = {num_labels} ทำให้ N < 3 → Ω2 ถูกบังคับเป็น 1 ตามนิยาม "
                "ค่า Ω′ ที่ได้จึงสูงเกินจริงและเทียบกับ |L| อื่นไม่ได้"
            ),
        })
    if sample_size < 3 * num_labels:
        warnings.append({
            "level": "warning",
            "text": (
                f"ข้อมูลมีเพียง {sample_size:,} คะแนนต่อ {num_labels} เกรด "
                f"(เฉลี่ย {sample_size / num_labels:.1f} คนต่อเกรด) "
                "ผลที่ได้ไวต่อคะแนนเดี่ยวมาก ใช้อ้างอิงอย่างระวัง"
            ),
        })
    if removed_count:
        warnings.append({
            "level": "info",
            "text": f"ตัดแถวที่ว่างหรือไม่ใช่ตัวเลขออก {removed_count:,} รายการก่อนคำนวณ",
        })
    if budget["mode"] == "refused":
        warnings.append({
            "level": "warning",
            "text": (
                f"ข้ามการค้นแบบ Exhaustive เพราะต้องประเมิน {budget['candidate_count']:,} "
                f"partitions ({format_duration(budget['estimated_seconds'])}) "
                "ผลนี้จึงไม่มี ground truth มายืนยันว่าเป็น global optimum"
            ),
        })
    elif budget["mode"] == "confirm" and not force_exhaustive:
        warnings.append({
            "level": "warning",
            "text": (
                f"ไม่ได้รัน Exhaustive ({budget['candidate_count']:,} partitions) "
                "ผลนี้เทียบกับ ground truth ไม่ได้"
            ),
        })
    return warnings


async def calculation_events(
    contents: bytes,
    filename: str,
    labels: str,
    force_exhaustive: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Run CPD and emit structured events for the live execution drawer."""
    started_at = time.perf_counter()
    step = 0

    def log_event(
        title: str,
        detail: str,
        level: str = "info",
        trace: str = "",
        anchor: bool = False,
    ) -> dict[str, Any]:
        """`trace` tags an entry with the method that produced it; `anchor`
        marks the one entry a results row should jump to."""
        nonlocal step
        step += 1
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        print(f"[CPD step {step:02d}] {title}: {detail}", flush=True)
        return {
            "type": "log",
            "step": step,
            "level": level,
            "title": title,
            "detail": detail,
            "elapsed_ms": elapsed_ms,
            "trace": trace,
            "anchor": anchor,
        }

    def number_list(values: list[float] | np.ndarray, limit: int = 40) -> str:
        """Format calculation inputs without flooding the trace for huge datasets."""
        formatted = [f"{float(value):g}" for value in values]
        if len(formatted) <= limit:
            return "[" + ", ".join(formatted) + "]"
        hidden_count = len(formatted) - limit
        return "[" + ", ".join(formatted[:limit]) + f", … +{hidden_count} values]"

    def omega_trace_events(method_label: str, result: Any, trace_key: str = ""):
        """Explain how every Ω component was derived for one partition."""
        cluster_bounds = result.cluster_bounds
        pvis = np.asarray(
            [high - low for high, low in cluster_bounds],
            dtype=float,
        )
        widest_pvi = float(pvis.max())
        deltas = np.asarray(
            [
                cluster_bounds[index][1] - cluster_bounds[index + 1][0]
                for index in range(result.N - 1)
            ],
            dtype=float,
        )
        gamma_u = upper_bound - cluster_bounds[0][0]
        gamma_l = cluster_bounds[-1][1] - lower_bound
        boundary_gaps = np.concatenate(([gamma_u], deltas, [gamma_l]))
        cluster_lines = [
            f"C{index + 1}: [{high:g}, {low:g}] → PVI{index + 1} = {high:g} - {low:g} = {high - low:g}"
            for index, (high, low) in enumerate(cluster_bounds)
        ]
        yield log_event(
            f"{method_label} · Partition/PVI",
            "\n".join(
                [
                    f"cuts = {result.cuts}; N = {result.N}",
                    *cluster_lines,
                    f"PVI = {number_list(pvis)}",
                    f"widest PVI = max(PVI) = {widest_pvi:g}",
                ]
            ),
            "formula",
            trace=trace_key,
        )

        named_gaps = [f"γu={gamma_u:g}"]
        named_gaps.extend(f"δ{index + 1}={value:g}" for index, value in enumerate(deltas))
        named_gaps.append(f"γl={gamma_l:g}")
        if result.Theta >= 1:
            raw_omega1 = result.theta / result.Theta
            omega1_formula = (
                f"Ω1 = clamp(θ[ตัวเล็ก] / Θ[ตัวใหญ่], 0, 1) "
                f"= clamp({result.theta} / {result.Theta}, 0, 1) "
                f"= {result.omega1:.6f} (raw={raw_omega1:.6f})"
            )
        else:
            omega1_formula = "Ω1 = 1 because Θ (Theta, ตัวใหญ่) = 0"
        yield log_event(
            f"{method_label} · Ω1 การจัดสรร label",
            "\n".join(
                [
                    "สัญลักษณ์: θ (theta, ตัวเล็ก) ≠ Θ (Theta, ตัวใหญ่)",
                    "θ = จำนวน labels ที่ไม่ได้ถูก assign",
                    "Θ = จำนวน gaps ที่กว้างอย่างน้อยเท่ากับ widest PVI",
                    f"θ (theta, ตัวเล็ก) = |L| - N = {num_labels} - {result.N} = {result.theta}",
                    f"boundary gaps = [{', '.join(named_gaps)}]",
                    f"condition: gap ≥ widest PVI = {widest_pvi:g}",
                    (
                        f"Θ (Theta, ตัวใหญ่) = count({number_list(boundary_gaps)} "
                        f"≥ {widest_pvi:g}) = {result.Theta}"
                    ),
                    omega1_formula,
                ]
            ),
            "formula",
            trace=trace_key,
        )

        if result.N >= 3:
            adjacent_gaps = np.abs(np.diff(values_desc))
            sorted_gaps = np.sort(adjacent_gaps)
            gap_count = result.N - 1
            min_gaps = sorted_gaps[:gap_count]
            max_gaps = sorted_gaps[::-1][:gap_count]
            d_min = float(min_gaps.sum())
            d_max = float(max_gaps.sum())
            delta_sum = float(deltas.sum())
            denominator = d_max - d_min
            if denominator == 0:
                omega2_formula = "Ω2 = 1 because D_max - D_min = 0"
            else:
                raw_omega2 = (delta_sum - d_min) / denominator
                omega2_formula = (
                    f"Ω2 = clamp((Σδ - D_min) / (D_max - D_min), 0, 1)\n"
                    f"   = clamp(({delta_sum:g} - {d_min:g}) / ({d_max:g} - {d_min:g}), 0, 1)\n"
                    f"   = {result.omega2:.6f} (raw={raw_omega2:.6f})"
                )
            omega2_lines = [
                f"m = N - 1 = {result.N} - 1 = {gap_count}",
                f"all adjacent score gaps = {number_list(adjacent_gaps)}",
                f"D_min gaps = {number_list(min_gaps)} → D_min = {d_min:g}",
                f"D_max gaps = {number_list(max_gaps)} → D_max = {d_max:g}",
                f"selected δ = {number_list(deltas)} → Σδ = {delta_sum:g}",
                omega2_formula,
            ]
        else:
            omega2_lines = [
                f"N = {result.N} < 3",
                "Ω2 = 1 by definition for fewer than 3 clusters",
            ]
        yield log_event(
            f"{method_label} · Ω2 ระยะห่างระหว่างกลุ่ม",
            "\n".join(omega2_lines),
            "formula",
            trace=trace_key,
        )

        if result.N >= 2:
            omega3_lines = [
                f"PVI = {number_list(pvis)}",
                f"σ = sample_std(PVI, ddof=1) = {result.sigma:.6f}",
                f"Ω3 = 1 / (1 + σ) = 1 / (1 + {result.sigma:.6f}) = {result.omega3:.6f}",
            ]
        else:
            omega3_lines = ["N < 2 → σ = 0 and Ω3 = 1"]
        yield log_event(
            f"{method_label} · Ω3 ความสม่ำเสมอของ PVI",
            "\n".join(omega3_lines),
            "formula",
            trace=trace_key,
        )

        yield log_event(
            f"{method_label} · Ω′ ผลลัพธ์สุดท้าย",
            (
                "Ω′ = Ω1 × Ω2 × Ω3\n"
                f"   = {result.omega1:.6f} × {result.omega2:.6f} × {result.omega3:.6f}\n"
                f"   = {result.omega_prime:.6f}"
            ),
            "formula",
            trace=trace_key,
            anchor=True,
        )

    try:
        yield log_event(
            "รับคำขอคำนวณ",
            f"ไฟล์ {filename} ขนาด {len(contents):,} bytes ถูกอัปโหลดเข้าสู่ระบบ",
        )

        df, file_type = await read_table(contents, filename)

        yield log_event(
            "อ่านไฟล์สำเร็จ",
            f"ตรวจพบไฟล์ {file_type}: {len(df):,} แถว, {len(df.columns):,} คอลัมน์",
            "success",
        )

        score_column, column_strategy = select_score_column(df)
        yield log_event(
            "เลือกคอลัมน์คะแนน",
            f"ใช้คอลัมน์ “{score_column}” ({column_strategy})",
        )

        values_desc, removed_count = extract_scores(df, score_column)
        cleanup_detail = (
            f"คงเหลือ {len(values_desc):,} คะแนน ช่วง {values_desc[-1]:g}–{values_desc[0]:g}"
        )
        if removed_count:
            cleanup_detail += f"; ตัดค่าที่ว่าง/ไม่ใช่ตัวเลขออก {removed_count:,} รายการ"
        yield log_event("ตรวจสอบและเรียงคะแนน", cleanup_detail, "success")

        grade_symbols = parse_labels(labels)
        num_labels = len(grade_symbols)
        check_label_budget(len(values_desc), num_labels)

        # k is |L| exactly — never min(|L|, n). check_label_budget guarantees
        # the data can carry the budget the user asked for.
        cluster_count = num_labels
        upper_bound, lower_bound = UPPER_BOUND, LOWER_BOUND
        yield log_event(
            "กำหนดค่าการแบ่งกลุ่ม",
            (
                f"labels=[{', '.join(grade_symbols)}], |L|=k={cluster_count} (คงที่ ไม่ sweep), "
                f"ขอบเขตคะแนน L={lower_bound:g}, U={upper_bound:g}"
            ),
        )

        budget = exhaustive_budget(len(values_desc), cluster_count)
        run_warnings = data_warnings(
            len(values_desc), num_labels, removed_count, budget, force_exhaustive,
        )
        for warning in run_warnings:
            yield log_event(
                "ข้อควรระวังของชุดข้อมูลนี้",
                warning["text"],
                "warning" if warning["level"] == "warning" else "info",
            )

        methods_evaluated: list[dict[str, Any]] = []

        yield log_event(
            "เริ่ม Dynamic Programming",
            f"ค้นหา 1-D k-means ที่มี SSE ต่ำที่สุดสำหรับ n={len(values_desc):,}, k={cluster_count}",
            "running",
        )
        algorithm_started = time.perf_counter()
        result_dp = await asyncio.to_thread(
            dp_best,
            values_desc,
            num_labels,
            upper_bound,
            lower_bound,
            cluster_count,
        )
        dp_elapsed_ms = (time.perf_counter() - algorithm_started) * 1000
        if result_dp:
            methods_evaluated.append(
                {
                    "name": "1-D k-means (Dynamic Programming)",
                    "trace": "DP",
                    "res": result_dp,
                }
            )
            yield log_event(
                "Dynamic Programming เสร็จสิ้น",
                (
                    f"cuts={result_dp.cuts}, Ω′={result_dp.omega_prime:.4f}, "
                    f"ใช้เวลา {dp_elapsed_ms:,.1f} ms"
                ),
                "success",
            )
            for trace_event in omega_trace_events("DP", result_dp, "DP"):
                yield trace_event

        candidate_count = budget["candidate_count"]
        candidate_text = f"{candidate_count:,} partitions = {budget['candidate_formula']}"
        run_exhaustive = budget["mode"] == "auto" or (
            budget["mode"] == "confirm" and force_exhaustive
        )

        result_exhaustive = None
        exhaustive_cancelled = False
        exhaustive_elapsed_ms = 0.0
        if not run_exhaustive:
            reason = (
                "เกินเพดานที่ระบบยอมให้รัน"
                if budget["mode"] == "refused"
                else "ผู้ใช้ไม่ได้ยืนยันให้รัน"
            )
            yield log_event(
                "ข้าม Exhaustive Search",
                (
                    f"{candidate_text} ≈ {format_duration(budget['estimated_seconds'])} — {reason}\n"
                    f"เพดานรันอัตโนมัติ = {EXHAUSTIVE_AUTO_LIMIT:,} partitions, "
                    f"เพดานเมื่อยืนยัน = {EXHAUSTIVE_CONFIRM_LIMIT:,} partitions"
                ),
                "warning",
            )
        else:
            exhaustive_level = "warning" if candidate_count > EXHAUSTIVE_AUTO_LIMIT else "running"
            yield log_event(
                "เริ่ม Exhaustive Search",
                (
                    f"ประเมินครบทุกความเป็นไปได้ {candidate_text}\n"
                    f"ประมาณการเวลา {format_duration(budget['estimated_seconds'])} "
                    f"(อัตราอ้างอิง {EXHAUSTIVE_RATE_PER_SEC:,} partitions/วินาที)"
                ),
                exhaustive_level,
            )
            algorithm_started = time.perf_counter()

            # The search runs in a worker thread that asyncio cannot cancel, so
            # give it a flag it checks itself. Without this a disconnected
            # client leaves a CPU-bound thread grinding for nobody.
            cancel_flag = threading.Event()
            progress = {"examined": 0}

            def monitor(examined: int) -> bool:
                progress["examined"] = examined
                return not cancel_flag.is_set()

            search_task = asyncio.create_task(
                asyncio.to_thread(
                    exhaustive_best_for_k,
                    values_desc,
                    cluster_count,
                    num_labels,
                    upper_bound,
                    lower_bound,
                    monitor=monitor,
                )
            )

            # When the client goes away Starlette cancels this generator, which
            # surfaces here as CancelledError/GeneratorExit at the next await or
            # yield. That is the only reliable disconnect signal during a
            # StreamingResponse — request.is_disconnected() competes for the same
            # receive channel Starlette's own listener already owns. Whatever
            # ends this block, the flag gets set so the worker thread stops
            # instead of grinding on for a client that no longer exists.
            try:
                while True:
                    finished, _ = await asyncio.wait({search_task}, timeout=HEARTBEAT_SECONDS)
                    if finished:
                        break
                    examined = progress["examined"]
                    search_elapsed = time.perf_counter() - algorithm_started
                    observed_rate = examined / search_elapsed if examined and search_elapsed else 0
                    remaining_text = (
                        format_duration((candidate_count - examined) / observed_rate)
                        if observed_rate
                        else "กำลังประเมิน"
                    )
                    yield log_event(
                        "Exhaustive Search กำลังทำงาน",
                        (
                            f"ตรวจแล้ว {examined:,} / {candidate_count:,} partitions "
                            f"({examined / candidate_count * 100:.1f}%)\n"
                            f"อัตราจริง {observed_rate:,.0f} partitions/วินาที · "
                            f"เหลืออีก {remaining_text}"
                        ),
                        "running",
                    )
                result_exhaustive = await search_task
            except SearchCancelled:
                exhaustive_cancelled = True
            finally:
                exhaustive_elapsed_ms = (time.perf_counter() - algorithm_started) * 1000
                if not search_task.done():
                    cancel_flag.set()
                    # Nobody will await this task now; consume its result so the
                    # loop does not report a never-retrieved exception.
                    search_task.add_done_callback(
                        lambda task: task.cancelled() or task.exception()
                    )
                    print(
                        f"[CPD] run abandoned; exhaustive search aborted after "
                        f"{progress['examined']:,} candidates",
                        flush=True,
                    )

        if exhaustive_cancelled:
            yield log_event(
                "Exhaustive Search ถูกยกเลิก",
                f"หยุดหลังตรวจไปบางส่วนใน {exhaustive_elapsed_ms:,.1f} ms",
                "warning",
            )
        if result_exhaustive:
            methods_evaluated.append(
                {
                    "name": "Exhaustive Search (Combinatorial)",
                    "trace": "Exhaustive",
                    "res": result_exhaustive,
                }
            )
            yield log_event(
                "Exhaustive Search เสร็จสิ้น",
                (
                    f"cuts={result_exhaustive.cuts}, Ω′={result_exhaustive.omega_prime:.4f}, "
                    f"ตรวจ {candidate_count:,} partitions ใน {exhaustive_elapsed_ms:,.1f} ms"
                ),
                "success",
            )
            for trace_event in omega_trace_events(
                "Exhaustive", result_exhaustive, "Exhaustive",
            ):
                yield trace_event

        if not methods_evaluated:
            raise CalculationFlowError(
                500,
                bilingual(
                    "คำนวณ partition ไม่สำเร็จเลยสักวิธี ตรวจสอบว่าข้อมูลมีคะแนน "
                    "ที่ไม่ซ้ำกันมากพอสำหรับจำนวนเกรดที่ระบุ",
                    "Failed to calculate CPD partitions with any method.",
                ),
            )

        methods_evaluated.sort(
            key=lambda method: method["res"].omega_prime,
            reverse=True,
        )
        best_match = methods_evaluated[0]
        result = best_match["res"]
        method_name = best_match["name"]
        yield log_event(
            "เลือกผลลัพธ์ที่ดีที่สุด",
            f"{method_name} ชนะด้วย Ω′={result.omega_prime:.4f}; cuts={result.cuts}",
            "success",
        )

        labeled, assigned_symbols = assign_labels(
            values_desc,
            result.cuts,
            grade_symbols,
        )
        # Build each cluster from its own contiguous span, not by matching on the
        # symbol string: two clusters can legitimately carry the same text and a
        # string match would count every record twice.
        cut_positions = [0, *sorted(int(cut) for cut in result.cuts), len(values_desc)]
        clusters = []
        for index, symbol in enumerate(assigned_symbols):
            segment = values_desc[cut_positions[index]:cut_positions[index + 1]]
            if len(segment) == 0:
                continue
            clusters.append(
                {
                    "grade": symbol,
                    "min": float(segment.min()),
                    "max": float(segment.max()),
                    "amount": int(len(segment)),
                }
            )

        labeled_records = [
            {"score": float(score), "label": grade} for score, grade in labeled
        ]
        cluster_summary = ", ".join(
            f"{cluster['grade']}={cluster['amount']} คน ({cluster['min']:g}–{cluster['max']:g})"
            for cluster in clusters
        )
        yield log_event("กำหนด label ให้แต่ละกลุ่ม", cluster_summary, "success")

        # Every field the comparison row renders — Ω′ alone is not a comparison.
        best_omega = methods_evaluated[0]["res"].omega_prime
        method_scores = [
            {
                "name": method["name"],
                "trace": method["trace"],
                "rank": index + 1,
                "omega_prime": round(method["res"].omega_prime, 4),
                "omega1": round(method["res"].omega1, 4),
                "omega2": round(method["res"].omega2, 4),
                "omega3": round(method["res"].omega3, 4),
                "sigma": round(method["res"].sigma, 4),
                "cluster_count": method["res"].N,
                "cuts": [int(cut) for cut in method["res"].cuts],
                "delta": round(method["res"].omega_prime - best_omega, 4),
                # Fraction of the best score, for the bar. Undefined when the
                # best is 0 — which happens for real (a DP-only run on tc1) —
                # so the client renders no bars at all in that case.
                "share": (
                    round(method["res"].omega_prime / best_omega, 6)
                    if best_omega > 0
                    else None
                ),
            }
            for index, method in enumerate(methods_evaluated)
        ]
        total_elapsed_ms = (time.perf_counter() - started_at) * 1000
        if result.N < 3:
            run_warnings.append({
                "level": "warning",
                "text": (
                    f"ผลลัพธ์มี N = {result.N} กลุ่ม (< 3) → Ω2 = 1 โดยนิยาม "
                    "ค่า Ω′ นี้จึงไม่สะท้อนระยะห่างระหว่างกลุ่มจริง"
                ),
            })

        response_data = {
            "omega_prime": round(result.omega_prime, 4),
            "omega1": round(result.omega1, 4),
            "omega2": round(result.omega2, 4),
            "omega3": round(result.omega3, 4),
            "sigma": round(result.sigma, 4),
            "clusters": clusters,
            "labeled_records": labeled_records,
            "n": len(values_desc),
            "method_name": method_name,
            "method_scores": method_scores,
            "warnings": run_warnings,
            # Run context: what produced these numbers, so the results screen
            # and the exported file can both answer "which run was this?".
            "source_filename": filename,
            "score_column": str(score_column),
            "column_strategy": column_strategy,
            "labels": ", ".join(grade_symbols),
            "label_budget": num_labels,
            "cluster_count": result.N,
            "cuts": [int(cut) for cut in result.cuts],
            "removed_rows": int(removed_count),
            "elapsed_ms": round(total_elapsed_ms, 1),
            "exhaustive": {
                "ran": bool(result_exhaustive),
                "cancelled": exhaustive_cancelled,
                "candidate_count": candidate_count,
                "candidate_formula": budget["candidate_formula"],
                "mode": budget["mode"],
            },
        }

        yield log_event(
            "การคำนวณเสร็จสมบูรณ์",
            f"เตรียมผลลัพธ์ {len(labeled_records):,} รายการใน {total_elapsed_ms:,.1f} ms",
            "complete",
        )
        yield {"type": "result", "data": response_data}

    except CalculationFlowError as exc:
        print(f"[CPD error] {exc.detail}", flush=True)
        yield {
            "type": "error",
            "status_code": exc.status_code,
            "message": exc.detail,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
        }
    except Exception as exc:
        print(traceback.format_exc(), flush=True)
        yield {
            "type": "error",
            "status_code": 500,
            "message": str(exc),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
        }


@app.post("/api/preflight")
async def preflight(
    file: UploadFile = File(...),
    labels: str = Form(...),
):
    """Quote the cost and the caveats of a run before anyone commits to it."""
    contents = await file.read()
    filename = file.filename or "uploaded-file"
    try:
        df, file_type = await read_table(contents, filename)
        score_column, column_strategy = select_score_column(df)
        values_desc, removed_count = extract_scores(df, score_column)
        grade_symbols = parse_labels(labels)
        check_label_budget(len(values_desc), len(grade_symbols))
    except CalculationFlowError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    num_labels = len(grade_symbols)
    budget = exhaustive_budget(len(values_desc), num_labels)
    return {
        "ok": True,
        "file_type": file_type,
        "source_filename": filename,
        "rows": int(len(df)),
        "n": int(len(values_desc)),
        "removed_rows": int(removed_count),
        "score_column": str(score_column),
        "column_strategy": column_strategy,
        "score_min": float(values_desc[-1]),
        "score_max": float(values_desc[0]),
        "labels": grade_symbols,
        "label_budget": num_labels,
        "exhaustive": {
            **budget,
            "estimated_text": format_duration(budget["estimated_seconds"]),
        },
        "warnings": data_warnings(
            len(values_desc), num_labels, removed_count, budget, force_exhaustive=True,
        ),
    }


@app.post("/api/calculate")
async def calculate_cpd(
    file: UploadFile = File(...),
    labels: str = Form(...),
    force_exhaustive: bool = Form(False),
):
    """Compatibility endpoint that returns the final calculation as JSON."""
    contents = await file.read()
    filename = file.filename or "uploaded-file"
    async for event in calculation_events(contents, filename, labels, force_exhaustive):
        if event["type"] == "result":
            return event["data"]
        if event["type"] == "error":
            raise HTTPException(
                status_code=event.get("status_code", 500),
                detail=event["message"],
            )
    raise HTTPException(status_code=500, detail="Calculation ended without a result.")


@app.post("/api/calculate/stream")
async def calculate_cpd_stream(
    file: UploadFile = File(...),
    labels: str = Form(...),
    force_exhaustive: bool = Form(False),
):
    """Stream calculation logs and the final result as newline-delimited JSON."""
    contents = await file.read()
    filename = file.filename or "uploaded-file"

    async def stream() -> AsyncIterator[str]:
        async for event in calculation_events(contents, filename, labels, force_exhaustive):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Export — the run has to leave the screen as a file
# ---------------------------------------------------------------------------

class ExportMethodScore(BaseModel):
    name: str
    omega_prime: float


class ExportCluster(BaseModel):
    grade: str
    min: float
    max: float
    amount: int


class ExportRecord(BaseModel):
    score: float
    label: str


class ExportManifest(BaseModel):
    source_filename: str = ""
    score_column: str = ""
    column_strategy: str = ""
    labels: str = ""
    label_budget: int = 0
    n: int = 0
    removed_rows: int = 0
    cluster_count: int = 0
    method_name: str = ""
    omega_prime: float = 0.0
    omega1: float = 0.0
    omega2: float = 0.0
    omega3: float = 0.0
    sigma: float = 0.0
    cuts: list[int] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    exhaustive_ran: bool = False
    exhaustive_candidates: int = 0
    method_scores: list[ExportMethodScore] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExportRequest(BaseModel):
    format: Literal["csv", "xlsx"]
    manifest: ExportManifest
    clusters: list[ExportCluster] = Field(default_factory=list)
    records: list[ExportRecord] = Field(default_factory=list)


# A leading =, +, -, or @ turns a spreadsheet cell into a formula. Grade symbols
# are free text, so neutralise them on the way out.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_FORMULA_TRIGGERS):
        return "'" + value
    return value


def export_filename(source: str, extension: str, stamp: str) -> str:
    stem = re.sub(r"\.(csv|xlsx|xls)$", "", source, flags=re.IGNORECASE) or "cpd"
    return f"{stem}-cpd-{stamp}.{extension}"


def content_disposition(filename: str) -> str:
    """ASCII fallback plus RFC 5987 name, so Thai filenames survive intact."""
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip("-._") or "cpd-result"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def export_frames(payload: ExportRequest, stamp: str) -> dict[str, pd.DataFrame]:
    """One builder for both formats, so CSV and XLSX can never disagree."""
    manifest = payload.manifest
    manifest_rows: list[tuple[str, Any]] = [
        ("generated_at", stamp),
        ("source_filename", manifest.source_filename),
        ("score_column", manifest.score_column),
        ("column_strategy", manifest.column_strategy),
        ("labels", manifest.labels),
        ("label_budget_|L|", manifest.label_budget),
        ("n", manifest.n),
        ("removed_rows", manifest.removed_rows),
        ("clusters_N", manifest.cluster_count),
        ("winning_method", manifest.method_name),
        ("omega_prime", manifest.omega_prime),
        ("omega1", manifest.omega1),
        ("omega2", manifest.omega2),
        ("omega3", manifest.omega3),
        ("sigma_ddof1", manifest.sigma),
        ("cuts", ", ".join(str(cut) for cut in manifest.cuts)),
        ("elapsed_ms", manifest.elapsed_ms),
        ("exhaustive_ran", manifest.exhaustive_ran),
        ("exhaustive_candidates", manifest.exhaustive_candidates),
    ]
    for score in manifest.method_scores:
        manifest_rows.append((f"omega_prime[{score.name}]", score.omega_prime))
    for index, warning in enumerate(manifest.warnings, start=1):
        manifest_rows.append((f"warning_{index}", warning))

    return {
        "Manifest": pd.DataFrame(
            [{"field": key, "value": safe_cell(value)} for key, value in manifest_rows]
        ),
        "Summary": pd.DataFrame(
            [
                {
                    "grade": safe_cell(cluster.grade),
                    "min": cluster.min,
                    "max": cluster.max,
                    "amount": cluster.amount,
                }
                for cluster in payload.clusters
            ]
        ),
        "Records": pd.DataFrame(
            [
                {"rank": index, "score": record.score, "grade": safe_cell(record.label)}
                for index, record in enumerate(payload.records, start=1)
            ]
        ),
    }


@app.post("/api/export")
async def export_result(payload: ExportRequest):
    """Return the run as a file: manifest, per-grade summary, per-record grades."""
    if not payload.records and not payload.clusters:
        raise HTTPException(
            status_code=400,
            detail=bilingual(
                "ไม่มีผลลัพธ์ให้ส่งออก กรุณาคำนวณก่อน",
                "Nothing to export: run a calculation first.",
            ),
        )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    frames = export_frames(payload, stamp)
    source = payload.manifest.source_filename or "cpd"

    if payload.format == "csv":
        buffer = io.StringIO()
        for title, frame in frames.items():
            buffer.write(f"# {title}\n")
            frame.to_csv(buffer, index=False)
            buffer.write("\n")
        # BOM so Excel opens the Thai grade symbols as UTF-8 rather than mojibake.
        body = ("﻿" + buffer.getvalue()).encode("utf-8")
        media_type = "text/csv; charset=utf-8"
        filename = export_filename(source, "csv", stamp)
    else:
        buffer_bytes = io.BytesIO()
        with pd.ExcelWriter(buffer_bytes, engine="openpyxl") as writer:
            for title, frame in frames.items():
                frame.to_excel(writer, sheet_name=title, index=False)
        body = buffer_bytes.getvalue()
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = export_filename(source, "xlsx", stamp)

    print(f"[CPD export] {filename} ({len(body):,} bytes)", flush=True)
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": content_disposition(filename)},
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
