"""Mark grasp and release frames by hand, one key press each.

Automatic segmentation from grasp aperture does not work for a wrap-around
cup grasp: the thumb stays about a cup-diameter from the fingers the whole
time, and it is usually occluded by the cup anyway. So we mark the events
directly. One pass over the clips, two keys per trial.

    python scripts/annotate.py data/episodes data/annotations.csv

Keys:
    space       play / pause
    . , (or arrows)  step one frame forward / back
    > <         jump 10 frames
    0 9         jump to the first / last frame
    l           replay this clip from the start
    g           mark GRASP at the current frame
    r           mark RELEASE at the current frame (closes the trial)
    u           undo the last mark in this clip
    x           reject this clip (logged, and skipped next run)
    n           next clip (saves)
    p           previous clip
    q           save and quit

Marks are written to CSV as you go, so quitting and resuming is safe:
already-annotated clips are skipped unless you pass --redo.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

WIN = "annotate - g=grasp  r=release  n=next  q=quit"


def load_csv(path: Path) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["clip"], []).append(
                (int(row["grasp_frame"]), int(row["release_frame"])))
    return out


def save_csv(path: Path, data: dict[str, list[tuple[int, int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "trial", "grasp_frame", "release_frame"])
        for clip in sorted(data):
            for i, (g, r) in enumerate(data[clip], 1):
                w.writerow([clip, i, g, r])


def load_rejects(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def save_rejects(path: Path, names: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(names)) + ("\n" if names else ""))


def read_frames(path: Path, max_width: int = 960) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] > max_width:
            s = max_width / fr.shape[1]
            fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    return frames


def draw(frame, clip_name, i, n, marks, pending, playing):
    img = frame.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.putText(img, f"{clip_name}   frame {i}/{n - 1}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    state = "PLAYING" if playing else "paused"
    pend = f"   grasp@{pending} -> press r" if pending is not None else ""
    cv2.putText(img, f"{state}   trials done: {len(marks)}{pend}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 220, 255) if pending is not None else (200, 200, 200), 1)
    cv2.putText(img, "g=grasp r=release u=undo l=replay 0=start x=reject n=next q=quit",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # timeline
    y = h - 18
    cv2.line(img, (10, y), (w - 10, y), (90, 90, 90), 3)
    def x_of(f):
        return int(10 + (w - 20) * (f / max(1, n - 1)))
    for g, r in marks:
        cv2.line(img, (x_of(g), y), (x_of(r), y), (0, 200, 0), 6)
    if pending is not None:
        cv2.line(img, (x_of(pending), y - 8), (x_of(pending), y + 8), (0, 220, 255), 2)
    cv2.line(img, (x_of(i), y - 10), (x_of(i), y + 10), (255, 255, 255), 2)
    return img


def annotate_clip(path: Path, marks: list[tuple[int, int]]) -> tuple[str, list]:
    frames = read_frames(path)
    if not frames:
        print(f"  !! no frames in {path.name}, skipping")
        return "next", marks

    n = len(frames)
    i, playing, pending = 0, False, None
    marks = list(marks)

    while True:
        cv2.imshow(WIN, draw(frames[i], path.name, i, n, marks, pending, playing))
        key = cv2.waitKey(33 if playing else 0) & 0xFF

        if playing and key == 255:
            i = min(i + 1, n - 1)
            if i == n - 1:
                playing = False
            continue

        if key in (ord(" "),):
            playing = not playing
        elif key in (ord("."), 83, ord("d")):
            i, playing = min(i + 1, n - 1), False
        elif key in (ord(","), 81, ord("a")):
            i, playing = max(i - 1, 0), False
        elif key == ord(">"):
            i, playing = min(i + 10, n - 1), False
        elif key == ord("<"):
            i, playing = max(i - 10, 0), False
        elif key == ord("0"):
            i, playing = 0, False
        elif key == ord("9"):
            i, playing = n - 1, False
        elif key == ord("l"):          # loop: back to start and play
            i, playing = 0, True
        elif key == ord("g"):
            pending = i
            print(f"    grasp @ {i}")
        elif key == ord("r"):
            if pending is None:
                print("    press g first")
            elif i <= pending:
                print("    release must come after grasp")
            else:
                marks.append((pending, i))
                print(f"    trial {len(marks)}: {pending} -> {i}")
                pending = None
        elif key == ord("u"):
            if pending is not None:
                pending = None
                print("    cleared pending grasp")
            elif marks:
                print(f"    removed trial {len(marks)}")
                marks.pop()
        elif key == ord("x"):
            print(f"    rejected {path.name}")
            return "reject", []
        elif key == ord("n"):
            return "next", marks
        elif key == ord("p"):
            return "prev", marks
        elif key == ord("q"):
            return "quit", marks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir", nargs="?", default="data/episodes")
    ap.add_argument("out_csv", nargs="?", default="data/annotations.csv")
    ap.add_argument("--redo", action="store_true",
                    help="also revisit clips that already have marks")
    args = ap.parse_args()

    clips = sorted(Path(args.clips_dir).glob("ep_*.mp4"))
    if not clips:
        raise SystemExit(f"no ep_*.mp4 in {args.clips_dir}")

    out = Path(args.out_csv)
    reject_path = out.with_name(out.stem + "_rejected.txt")
    data = load_csv(out)
    rejected = load_rejects(reject_path)
    print(f"{len(clips)} clips, {len(data)} annotated, {len(rejected)} rejected")

    todo = clips if args.redo else [
        c for c in clips if c.name not in data and c.name not in rejected]
    if not todo:
        print("everything is annotated already; pass --redo to revisit")
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    idx = 0
    while 0 <= idx < len(todo):
        clip = todo[idx]
        print(f"\n[{idx + 1}/{len(todo)}] {clip.name}")
        action, marks = annotate_clip(clip, data.get(clip.name, []))

        if action == "reject":
            rejected.add(clip.name)
            data.pop(clip.name, None)
        else:
            rejected.discard(clip.name)
            if marks:
                data[clip.name] = marks
            elif clip.name in data:
                del data[clip.name]

        save_csv(out, data)
        save_rejects(reject_path, rejected)

        if action == "quit":
            break
        idx += 1 if action in ("next", "reject") else -1
        idx = max(0, idx)

    cv2.destroyAllWindows()
    total = sum(len(v) for v in data.values())
    print(f"\nsaved {out}: {len(data)} clips, {total} trials")
    if rejected:
        print(f"rejected {len(rejected)} clips -> {reject_path}")


if __name__ == "__main__":
    main()
