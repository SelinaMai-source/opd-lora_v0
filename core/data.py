import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from core.utils import ensure_dir


@dataclass
class Example:
    instruction: str
    input: str
    output: str


@dataclass
class Segment:
    segment_id: int
    segment_name: str
    train: List[Example]
    eval: List[Example]


@dataclass
class ContinualStream:
    benchmark: str
    version: str
    stream: List[Segment]


@dataclass(frozen=True)
class ProcessedStreamSpec:
    stream_name: str
    aliases: Tuple[str, ...]
    processed_file: str
    split_name: str
    benchmark_name: str
    version: str


KNOWN_PROCESSED_STREAMS: Tuple[ProcessedStreamSpec, ...] = (
    ProcessedStreamSpec(
        stream_name="instrdialog",
        aliases=("instrdialog", "citb_dialogue", "cl_dialogue_tasks", "dialogue"),
        processed_file="citb_cl_dialogue_tasks_train50_eval10.json",
        split_name="cl_dialogue_tasks",
        benchmark_name="CITB-InstrDialog",
        version="citb_instrdialog_train50_eval10_v1",
    ),
    ProcessedStreamSpec(
        stream_name="instrdialog++",
        aliases=("instrdialog++", "instrdialogpp", "citb_38_random", "38_random_tasks", "random38"),
        processed_file="citb_cl_38_random_tasks_train50_eval10.json",
        split_name="38_random_tasks",
        benchmark_name="CITB-InstrDialog++",
        version="citb_instrdialogpp_train50_eval10_v1",
    ),
    ProcessedStreamSpec(
        stream_name="trace",
        aliases=("trace", "trace_rcl", "trace_cl", "trace_benchmark"),
        processed_file="trace_cl_tasks_train50_eval10.json",
        split_name="trace_cl_tasks",
        benchmark_name="TRACE",
        version="trace_train50_eval10_v1",
    ),
    ProcessedStreamSpec(
        stream_name="multiwoz_nlg",
        aliases=("multiwoz_nlg", "multiwoz", "mwoz_nlg", "multiwoz_nlg_domains"),
        processed_file="multiwoz_nlg_cl_domains_train50_eval10.json",
        split_name="multiwoz_nlg_domains",
        benchmark_name="MultiWOZ-NLG",
        version="multiwoz_nlg_domains_train50_eval10_v1",
    ),
)


def _stream_alias_index() -> Dict[str, ProcessedStreamSpec]:
    out: Dict[str, ProcessedStreamSpec] = {}
    for spec in KNOWN_PROCESSED_STREAMS:
        for alias in (spec.stream_name, *spec.aliases):
            out[alias.strip().lower()] = spec
    return out


STREAM_ALIAS_TO_SPEC = _stream_alias_index()


def get_processed_stream_spec(stream_name: str) -> Optional[ProcessedStreamSpec]:
    if not stream_name:
        return None
    return STREAM_ALIAS_TO_SPEC.get(stream_name.strip().lower())


def list_known_processed_streams() -> List[Dict[str, str]]:
    return [
        {
            "stream_name": spec.stream_name,
            "processed_file": spec.processed_file,
            "split_name": spec.split_name,
            "benchmark_name": spec.benchmark_name,
            "version": spec.version,
        }
        for spec in KNOWN_PROCESSED_STREAMS
    ]


def _parse_example(obj: Dict[str, Any]) -> Example:
    for k in ["instruction", "input", "output"]:
        if k not in obj:
            raise ValueError(f"Example missing key '{k}': {obj}")
    return Example(
        instruction=str(obj["instruction"]),
        input=str(obj.get("input", "")),
        output=str(obj["output"]),
    )


def _parse_stream_json(path: str) -> ContinualStream:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "stream" not in raw or not isinstance(raw["stream"], list):
        raise ValueError(f"Invalid stream json: missing list field 'stream' in {path}")

    segments: List[Segment] = []
    for seg in raw["stream"]:
        if "segment_id" not in seg:
            raise ValueError(f"Segment missing 'segment_id': {seg}")
        if "segment_name" not in seg:
            raise ValueError(f"Segment missing 'segment_name': {seg}")
        train = [_parse_example(x) for x in seg.get("train", [])]
        ev = [_parse_example(x) for x in seg.get("eval", [])]
        if not train or not ev:
            raise ValueError(
                f"Each segment must include non-empty train & eval lists. Bad segment: {seg.get('segment_id')}"
            )
        segments.append(
            Segment(
                segment_id=int(seg["segment_id"]),
                segment_name=str(seg["segment_name"]),
                train=train,
                eval=ev,
            )
        )

    return ContinualStream(
        benchmark=str(raw.get("benchmark", "unknown")),
        version=str(raw.get("version", "unknown")),
        stream=segments,
    )


def load_continual_stream(
    *,
    mode: str,
    sample_stream_path: Optional[str],
    processed_stream_dir: Optional[str],
    processed_stream_file: str = "",
    processed_stream_name: str = "",
    auto_prepare_processed: bool = False,
    raw_citb_root: Optional[str] = None,
    seed: int = 10,
    processed_stream_train_instances_per_task: int = 50,
    processed_stream_eval_instances_per_task: int = 10,
    processed_stream_limit_tasks: int = -1,
    max_segments: int = -1,
    max_train_examples_per_segment: int = -1,
    max_eval_examples_per_segment: int = -1,
) -> ContinualStream:
    """
    Unified stream loader.

    - debug 模式：读取 data/sample/mock_stream.json
    - baseline/ours：读取 data/processed 下的 CITB 处理后流式文件（JSON）
    """

    if mode == "debug":
        if not sample_stream_path:
            raise ValueError("debug mode requires sample_stream_path")
        stream = _parse_stream_json(sample_stream_path)
    else:
        if not processed_stream_dir:
            raise ValueError("baseline/ours mode requires processed_stream_dir")
        stream_path = prepare_processed_stream_path(
            processed_stream_dir=processed_stream_dir,
            processed_stream_file=processed_stream_file,
            processed_stream_name=processed_stream_name,
            auto_prepare_processed=auto_prepare_processed,
            raw_citb_root=raw_citb_root,
            seed=seed,
            max_train_instances_per_task=processed_stream_train_instances_per_task,
            max_eval_instances_per_task=processed_stream_eval_instances_per_task,
            limit_tasks=processed_stream_limit_tasks,
        )
        stream = _parse_stream_json(stream_path)

    stream = truncate_stream(
        stream,
        max_segments=max_segments,
        max_train_examples_per_segment=max_train_examples_per_segment,
        max_eval_examples_per_segment=max_eval_examples_per_segment,
    )
    return stream


def truncate_stream(
    stream: ContinualStream,
    *,
    max_segments: int,
    max_train_examples_per_segment: int,
    max_eval_examples_per_segment: int,
) -> ContinualStream:
    segments = stream.stream
    if max_segments is not None and max_segments > 0:
        segments = segments[: max_segments]

    new_segments: List[Segment] = []
    for seg in segments:
        tr = seg.train
        ev = seg.eval
        if max_train_examples_per_segment is not None and max_train_examples_per_segment > 0:
            tr = tr[: max_train_examples_per_segment]
        if max_eval_examples_per_segment is not None and max_eval_examples_per_segment > 0:
            ev = ev[: max_eval_examples_per_segment]
        new_segments.append(
            Segment(segment_id=seg.segment_id, segment_name=seg.segment_name, train=tr, eval=ev)
        )

    return ContinualStream(benchmark=stream.benchmark, version=stream.version, stream=new_segments)


def resolve_processed_stream_path(
    processed_stream_dir: str,
    processed_stream_file: str = "",
    processed_stream_name: str = "",
) -> str:
    """
    Find a processed stream json path.

    - If processed_stream_file is provided, use it (relative to processed_stream_dir if not absolute)
    - Else if `processed_stream_name` matches a known benchmark alias, map it to the canonical file
    - Otherwise, try to auto-detect a single *.json file inside processed_stream_dir
    """

    d = Path(processed_stream_dir)
    if not d.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {processed_stream_dir}. "
            f"Please generate processed stream under data/processed/."
        )

    if processed_stream_file:
        p = Path(processed_stream_file)
        if not p.is_absolute():
            p = d / p
        if not p.exists():
            raise FileNotFoundError(f"Processed stream file not found: {str(p)}")
        return str(p)

    spec = get_processed_stream_spec(processed_stream_name)
    if spec is not None:
        p = d / spec.processed_file
        if not p.exists():
            raise FileNotFoundError(
                f"Processed stream for '{processed_stream_name}' not found: {p}. "
                "You can enable auto_prepare_processed in config to build it from raw CITB."
            )
        return str(p)

    candidates = sorted(list(d.glob("*.json")))
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No processed stream json found in {processed_stream_dir}. "
            f"Expected a single *.json. See data/processed/README.md."
        )
    raise FileExistsError(
        f"Multiple processed stream json files found in {processed_stream_dir}: "
        f"{[c.name for c in candidates]}. Please set processed_stream_file in config."
    )


def prepare_processed_stream_path(
    *,
    processed_stream_dir: str,
    processed_stream_file: str = "",
    processed_stream_name: str = "",
    auto_prepare_processed: bool = False,
    raw_citb_root: Optional[str] = None,
    seed: int = 10,
    max_train_instances_per_task: int = 50,
    max_eval_instances_per_task: int = 10,
    limit_tasks: int = -1,
) -> str:
    try:
        return resolve_processed_stream_path(
            processed_stream_dir=processed_stream_dir,
            processed_stream_file=processed_stream_file,
            processed_stream_name=processed_stream_name,
        )
    except FileNotFoundError:
        if processed_stream_file or not auto_prepare_processed:
            raise

    spec = get_processed_stream_spec(processed_stream_name)
    if spec is None:
        raise FileNotFoundError(
            f"Could not resolve processed stream '{processed_stream_name}'. "
            f"Known options: {[x['stream_name'] for x in list_known_processed_streams()]}"
        )

    out_path = Path(processed_stream_dir) / spec.processed_file
    preprocess_citb_raw_to_processed(
        raw_root=str(raw_citb_root or Path("data/raw/citb")),
        processed_out_path=str(out_path),
        benchmark_name=spec.benchmark_name,
        version=spec.version,
        split_name=spec.split_name,
        seed=seed,
        max_train_instances_per_task=max_train_instances_per_task,
        max_eval_instances_per_task=max_eval_instances_per_task,
        limit_tasks=limit_tasks,
    )
    return str(out_path)


def preprocess_citb_raw_to_processed(
    raw_root: str,
    processed_out_path: str,
    *,
    benchmark_name: str = "CITB",
    version: str = "unknown",
    split_name: str = "cl_dialogue_tasks",
    seed: int = 10,
    max_train_instances_per_task: int = 50,
    max_eval_instances_per_task: int = 10,
    limit_tasks: int = -1,
) -> None:
    """
    Preprocessing scaffold (NOT a fake implementation).

    This function is the intended extension point to convert raw CITB data in
    data/raw/citb/ into the unified continual stream JSON format under data/processed/.

    Current behavior:
      - validates input directories exist
      - creates output directory
      - raises an informative error describing what to implement

    Why keep it here?
      - Repo structure constraint: preprocessing logic must live in core/data.py.
    """

    raw_root_p = Path(raw_root)
    if not raw_root_p.exists():
        raise FileNotFoundError(
            f"Raw CITB root not found: {raw_root}. "
            f"Place raw data under data/raw/citb/ (see data/raw/README.md)."
        )

    ensure_dir(str(Path(processed_out_path).parent))

    citb_root = Path(raw_root)
    tasks_dir = citb_root / "data" / "tasks"
    splits_dir = citb_root / "data" / "splits" / "CIT_splits"

    split_txt_path = splits_dir / f"{split_name}.txt"
    if not split_txt_path.exists():
        raise FileNotFoundError(f"CITB split txt not found: {split_txt_path}")
    if not tasks_dir.exists():
        raise FileNotFoundError(f"CITB tasks dir not found: {tasks_dir}")

    rng = random.Random(int(seed))

    task_order: List[str] = []
    with split_txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            task_order.append(t)
            if int(limit_tasks) > 0 and len(task_order) >= int(limit_tasks):
                break
    if not task_order:
        raise ValueError(f"No tasks found in split file: {split_txt_path}")

    def first_definition(def_obj: Any) -> str:
        if isinstance(def_obj, list) and def_obj:
            return str(def_obj[0])
        if def_obj is None:
            return ""
        return str(def_obj)

    def to_outputs(output_obj: Any) -> List[str]:
        # CITB task json typically uses `output: [str, ...]` inside each instance.
        if output_obj is None:
            return [""]
        if isinstance(output_obj, list):
            return [str(x) for x in output_obj]
        return [str(output_obj)]

    def instance_to_examples(task_instruction: str, inst: Dict[str, Any]) -> List[Dict[str, str]]:
        in_text = str(inst.get("input", ""))
        outs = to_outputs(inst.get("output"))
        return [
            {"instruction": task_instruction, "input": in_text, "output": o}
            for o in outs
        ]

    segments: List[Dict[str, Any]] = []
    for seg_id, task_name in enumerate(task_order):
        task_json_path = tasks_dir / f"{task_name}.json"
        if not task_json_path.exists():
            raise FileNotFoundError(f"Task JSON not found for {task_name}: {task_json_path}")

        with task_json_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        instruction = first_definition(obj.get("Definition"))
        instances = obj.get("Instances") or []
        if not isinstance(instances, list) or not instances:
            raise ValueError(f"No Instances found in task json: {task_json_path}")

        max_eval = max(0, int(max_eval_instances_per_task))
        max_train = max(0, int(max_train_instances_per_task))

        # Mirror `train_dev_test_split_by_task(..., continual=False)` behavior:
        # - test: first max_eval instances
        # - dev: next max_eval instances (ignored in our processed format)
        # - remaining after max_eval*2: shuffle, then take max_train instances
        test_instances = instances[:max_eval]
        remaining = instances[max_eval * 2 :]
        rng.shuffle(remaining)
        train_instances = remaining[:max_train]

        train_examples: List[Dict[str, str]] = []
        for inst in train_instances:
            train_examples.extend(instance_to_examples(instruction, inst))

        eval_examples: List[Dict[str, str]] = []
        for inst in test_instances:
            eval_examples.extend(instance_to_examples(instruction, inst))

        segments.append(
            {
                "segment_id": seg_id,
                "segment_name": task_name,
                "train": train_examples,
                "eval": eval_examples,
            }
        )

        print(
            f"[{seg_id:03d}] {task_name}: "
            f"train_instances={len(train_instances)} train_examples={len(train_examples)} "
            f"eval_instances={len(test_instances)} eval_examples={len(eval_examples)}"
        )

    out = {"benchmark": benchmark_name, "version": version, "stream": segments}
    ensure_dir(str(Path(processed_out_path).parent))
    Path(processed_out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {processed_out_path}")

