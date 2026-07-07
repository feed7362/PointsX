#!/usr/bin/env python3
"""Desktop dataset viewer — CustomTkinter UI."""

from __future__ import annotations

import io
import os
import sys
import threading
import tkinter as tk
from datetime import datetime
from functools import partial
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

from repository import DatasetRepository

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

# FitMeasure light palette
BG = "#f4f6fb"
SURFACE = "#ffffff"
PHOTO_BG = "#eef1f7"
TEXT = "#1a1f2a"
MUTED = "#5c6578"
ACCENT = "#2f6fed"
ACCENT_SOFT = "#e8efff"
DANGER = "#c62828"
DANGER_SOFT = "#fde8e8"
SELECTED = "#dce8ff"
CORNER = 16


def format_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def format_date_short(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%m/%d %H:%M")
    except ValueError:
        return iso[:16] if iso else "—"


def sex_label(sex: str | None) -> str:
    return sex.capitalize() if sex else "—"


def bool_label(value) -> str:
    return "Yes" if value else "No"


def fit_image(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    out = image.copy()
    out.thumbnail((max(1, max_w), max(1, max_h)), Image.Resampling.LANCZOS)
    return out


class ConfirmDeleteDialog(ctk.CTkToplevel):
    """Second confirmation: user must type DELETE."""

    def __init__(self, master: ctk.CTk, submission_id: str) -> None:
        super().__init__(master)
        self.result = False
        self.title("Final confirmation")
        self.geometry("440x220")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        body = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=CORNER)
        body.pack(fill="both", expand=True, padx=16, pady=16)

        ctk.CTkLabel(
            body,
            text="This permanently deletes the record and encrypted photos.",
            font=ctk.CTkFont(size=14),
            text_color=TEXT,
            wraplength=380,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(16, 8))

        ctk.CTkLabel(
            body,
            text=f"ID: {submission_id}",
            font=ctk.CTkFont(family="Menlo", size=12),
            text_color=MUTED,
            wraplength=380,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))

        ctk.CTkLabel(
            body,
            text='Type DELETE to confirm:',
            font=ctk.CTkFont(size=13),
            text_color=TEXT,
        ).pack(anchor="w", padx=16)

        self.entry = ctk.CTkEntry(body, placeholder_text="DELETE", height=36, corner_radius=10)
        self.entry.pack(fill="x", padx=16, pady=(6, 12))
        self.entry.bind("<Return>", lambda _e: self._confirm())

        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkButton(
            actions,
            text="Cancel",
            fg_color=PHOTO_BG,
            hover_color="#dde3ef",
            text_color=TEXT,
            corner_radius=10,
            command=self._cancel,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            actions,
            text="Delete permanently",
            fg_color=DANGER,
            hover_color="#a31f1f",
            corner_radius=10,
            command=self._confirm,
        ).pack(side="right")

        self.after(100, self.entry.focus_set)

    def _cancel(self) -> None:
        self.result = False
        self.destroy()

    def _confirm(self) -> None:
        if self.entry.get().strip() == "DELETE":
            self.result = True
            self.destroy()
        else:
            messagebox.showerror("Confirmation failed", 'You must type exactly: DELETE')


class DatasetViewerApp(ctk.CTk):
    def __init__(self, repo: DatasetRepository) -> None:
        super().__init__()
        self.repo = repo
        self.items: list[dict] = []
        self._selected_id: str | None = None
        self._selected_index: int | None = None
        self._load_token = 0
        self._resize_job: str | None = None
        self._wrap_job: str | None = None
        self._last_refit_size: tuple[int, int] = (0, 0)
        self._last_photo_bytes: dict[str, bytes] = {}
        self._list_row_frames: list[ctk.CTkFrame] = []
        self._photo_labels: dict[str, ctk.CTkLabel] = {}
        self._photo_title_labels: dict[str, ctk.CTkLabel] = {}
        self._photo_images: dict[str, ctk.CTkImage] = {}

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("Dataset Viewer")
        self.minsize(1024, 680)
        self.geometry("1280x860")
        self.configure(fg_color=BG)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_body()
        self.bind("<Configure>", self._on_resize)
        self.after(50, self.refresh_list)
        self.after(300, self._update_wraplengths)

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=CORNER, height=64)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text="Dataset Viewer",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, padx=20, pady=16, sticky="w")

        ctk.CTkLabel(
            header,
            text="FitMeasure · local operator tool",
            font=ctk.CTkFont(size=13),
            text_color=MUTED,
        ).grid(row=0, column=1, padx=8, pady=16, sticky="w")

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(body, fg_color=SURFACE, corner_radius=CORNER, width=300)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        sidebar.grid_propagate(False)

        top = ctk.CTkFrame(sidebar, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(14, 8))
        ctk.CTkLabel(top, text="Submissions", font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT).pack(
            side="left"
        )
        self.count_label = ctk.CTkLabel(
            top,
            text="—",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT,
            fg_color=ACCENT_SOFT,
            corner_radius=12,
            padx=10,
            pady=4,
        )
        self.count_label.pack(side="right")

        ctk.CTkButton(
            sidebar,
            text="↻  Refresh",
            height=36,
            corner_radius=10,
            fg_color=PHOTO_BG,
            hover_color="#dde3ef",
            text_color=TEXT,
            command=self.refresh_list,
        ).pack(fill="x", padx=14, pady=(0, 8))

        self.list_scroll = ctk.CTkScrollableFrame(
            sidebar,
            fg_color=PHOTO_BG,
            corner_radius=12,
            scrollbar_button_color=ACCENT_SOFT,
            scrollbar_button_hover_color=ACCENT,
        )
        self.list_scroll.pack(fill="both", expand=True, padx=14, pady=(0, 8))

        self.status_label = ctk.CTkLabel(
            sidebar,
            text="Starting…",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            wraplength=250,
            justify="left",
        )
        self.status_label.pack(fill="x", padx=14, pady=(0, 14))

        self._enable_panel_scroll(self.list_scroll)

        # Detail
        self.detail_scroll = ctk.CTkScrollableFrame(
            body,
            fg_color=SURFACE,
            corner_radius=CORNER,
            scrollbar_button_color=ACCENT_SOFT,
            scrollbar_button_hover_color=ACCENT,
        )
        self.detail_scroll.grid(row=0, column=1, sticky="nsew")

        self._enable_panel_scroll(self.detail_scroll)

        self.detail_root = ctk.CTkFrame(self.detail_scroll, fg_color="transparent")
        self.detail_root.pack(fill="both", expand=True, anchor="nw")

        self.empty_label = ctk.CTkLabel(
            self.detail_root,
            text="Select a submission from the list",
            font=ctk.CTkFont(size=15),
            text_color=MUTED,
        )
        self.empty_label.pack(pady=80)

        self.detail_content = ctk.CTkFrame(self.detail_root, fg_color="transparent")

        id_card = ctk.CTkFrame(self.detail_content, fg_color=PHOTO_BG, corner_radius=CORNER)
        id_card.pack(fill="x", pady=(4, 12))
        ctk.CTkLabel(
            id_card,
            text="SUBMISSION ID",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=18, pady=(14, 4))
        self.id_label = ctk.CTkLabel(
            id_card,
            text="",
            font=ctk.CTkFont(family="Menlo", size=14, weight="bold"),
            text_color=ACCENT,
            wraplength=700,
            justify="left",
            anchor="w",
        )
        self.id_label.pack(fill="x", padx=18, pady=(0, 14))

        photos = ctk.CTkFrame(self.detail_content, fg_color="transparent")
        photos.pack(fill="x", pady=(0, 8))
        photos.grid_columnconfigure(0, weight=1)
        photos.grid_columnconfigure(1, weight=1)

        self.front_photo_frame, self.front_photo_label, self.front_title = self._photo_panel(photos, "Front", 0)
        self.side_photo_frame, self.side_photo_label, self.side_title = self._photo_panel(photos, "Side", 1)
        self._photo_labels = {"front": self.front_photo_label, "side": self.side_photo_label}
        self._photo_title_labels = {"front": self.front_title, "side": self.side_title}

        self.cards_frame = ctk.CTkFrame(self.detail_content, fg_color="transparent")
        self.cards_frame.pack(fill="x", pady=(4, 8))
        self.cards_frame.grid_columnconfigure(0, weight=1)
        self.cards_frame.grid_columnconfigure(1, weight=1)

        self.delete_btn = ctk.CTkButton(
            self.detail_content,
            text="Delete submission…",
            height=40,
            corner_radius=10,
            fg_color=DANGER_SOFT,
            hover_color="#f5c4c4",
            text_color=DANGER,
            border_width=1,
            border_color="#f0b4b4",
            command=self._delete_selected,
        )

    def _photo_panel(self, parent: ctk.CTkFrame, title: str, column: int) -> tuple[ctk.CTkFrame, ctk.CTkLabel, ctk.CTkLabel]:
        outer = ctk.CTkFrame(parent, fg_color=PHOTO_BG, corner_radius=CORNER)
        outer.grid(row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0), pady=4)

        title_label = ctk.CTkLabel(
            outer,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        )
        title_label.pack(anchor="w", padx=14, pady=(12, 6))

        img_label = ctk.CTkLabel(outer, text="No photo", text_color=MUTED, fg_color=PHOTO_BG, corner_radius=12)
        img_label.pack(padx=12, pady=(0, 12))
        return outer, img_label, title_label

    def _photo_max_size(self) -> tuple[int, int]:
        self.update_idletasks()
        w = max(self.detail_scroll.winfo_width() - 80, 400)
        h = max(self.winfo_height() - 280, 280)
        return max(160, (w - 40) // 2), max(180, int(h * 0.38))

    def _render_list(self) -> None:
        for child in self.list_scroll.winfo_children():
            child.destroy()
        self._list_row_frames.clear()

        for index, item in enumerate(self.items):
            selected = index == self._selected_index
            row = ctk.CTkFrame(
                self.list_scroll,
                fg_color=SELECTED if selected else "transparent",
                corner_radius=10,
                cursor="hand2",
            )
            row.pack(fill="x", pady=3)
            row.bind("<Button-1>", lambda _e, i=index: self._select_index(i))

            date_short = format_date_short(item.get("created_at"))
            meta = f"{item.get('age_years', '—')}y · {sex_label(item.get('sex'))}"

            id_lbl = ctk.CTkLabel(
                row,
                text=f"{item['id'][:8]}…",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=TEXT,
                anchor="w",
                wraplength=250,
                justify="left",
            )
            id_lbl.pack(anchor="w", padx=10, pady=(8, 0))
            id_lbl.bind("<Button-1>", lambda _e, i=index: self._select_index(i))

            meta_lbl = ctk.CTkLabel(
                row,
                text=f"{date_short}\n{meta}",
                font=ctk.CTkFont(size=11),
                text_color=MUTED,
                anchor="w",
                wraplength=250,
                justify="left",
            )
            meta_lbl.pack(anchor="w", padx=10, pady=(2, 8))
            meta_lbl.bind("<Button-1>", lambda _e, i=index: self._select_index(i))

            self._list_row_frames.append(row)

        self.after_idle(self._refresh_scroll_regions)
        self._schedule_wraplengths()

    def _enable_panel_scroll(self, panel: ctk.CTkScrollableFrame) -> None:
        """Route trackpad / mouse-wheel to the panel under the cursor."""
        canvas = panel._parent_canvas

        def _wheel(event: tk.Event) -> str:
            self._refresh_scroll_regions()
            first, last = canvas.yview()
            if first == 0.0 and last == 1.0:
                return "break"
            if sys.platform == "darwin":
                delta = int(-event.delta)
            elif sys.platform.startswith("win"):
                delta = int(-event.delta / 6)
            else:
                delta = -1 if event.num == 4 else 1
            canvas.yview("scroll", delta, "units")
            return "break"

        def _bind(_event: tk.Event) -> None:
            canvas.bind("<MouseWheel>", _wheel, add="+")
            panel.bind("<MouseWheel>", _wheel, add="+")

        def _unbind(_event: tk.Event) -> None:
            canvas.unbind("<MouseWheel>")
            panel.unbind("<MouseWheel>")

        panel.bind("<Enter>", _bind)
        panel.bind("<Leave>", _unbind)

    def _select_index(self, index: int) -> None:
        prev = self._selected_index
        self._selected_index = index
        item = self.items[index]
        self._selected_id = item["id"]
        if prev is not None and 0 <= prev < len(self._list_row_frames):
            self._list_row_frames[prev].configure(fg_color="transparent")
        if 0 <= index < len(self._list_row_frames):
            self._list_row_frames[index].configure(fg_color=SELECTED)
        self._show_detail(item["id"])

    def refresh_list(self) -> None:
        self.status_label.configure(text="Loading…", text_color=MUTED)

        def worker() -> None:
            try:
                items = self.repo.list_submissions()
                self.after(0, partial(self._apply_list, items, None))
            except Exception as exc:
                self.after(0, partial(self._apply_list, [], str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_list(self, items: list[dict], error: str | None) -> None:
        self.items = items
        if error:
            self.status_label.configure(text=f"Error: {error}", text_color=DANGER)
            self.count_label.configure(text="—")
            return

        count = len(self.items)
        self.count_label.configure(text=f"{count} record{'s' if count != 1 else ''}")
        if count == 0:
            self._selected_id = None
            self._selected_index = None
            self.status_label.configure(text="No submissions yet.", text_color=MUTED)
            self._render_list()
            self._show_empty_detail()
            return

        self.status_label.configure(text="Click a row to view details.", text_color=MUTED)
        if self._selected_id:
            self._selected_index = next(
                (i for i, it in enumerate(self.items) if it["id"] == self._selected_id),
                None,
            )
        self._render_list()

        self._schedule_wraplengths()

    def _show_empty_detail(self) -> None:
        self.detail_content.pack_forget()
        self.empty_label.pack(pady=80)
        self._refresh_scroll_regions()

    def _show_detail(self, submission_id: str) -> None:
        self.empty_label.pack_forget()
        self.detail_content.pack(fill="x", anchor="nw")
        self.delete_btn.pack(fill="x", pady=(8, 4))
        self.id_label.configure(text=submission_id)
        self._update_wraplengths()
        self._clear_cards()
        self._last_photo_bytes.clear()
        self._set_photo_placeholder("front", "Loading…")
        self._set_photo_placeholder("side", "Loading…")
        self._refresh_scroll_regions()

        self._load_token += 1
        token = self._load_token

        def worker() -> None:
            try:
                detail = self.repo.get_submission(submission_id)
                front = self.repo.get_photo_bytes(submission_id, "front")
                side = self.repo.get_photo_bytes(submission_id, "side")
                self.after(0, partial(self._apply_detail, token, detail, front, side))
            except Exception as exc:
                self.after(0, partial(self._apply_error, token, str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_error(self, token: int, message: str) -> None:
        if token != self._load_token:
            return
        self._set_photo_placeholder("front", "Failed")
        self._set_photo_placeholder("side", "Failed")
        self._clear_cards()
        ctk.CTkLabel(
            self.cards_frame,
            text=message,
            text_color=DANGER,
            wraplength=760,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        self._refresh_scroll_regions()

    def _apply_detail(self, token: int, detail: dict, front: bytes, side: bytes) -> None:
        if token != self._load_token:
            return

        self._show_photo("front", "Front", front)
        self._show_photo("side", "Side", side)
        self._clear_cards()

        self._info_card(
            "Submission",
            [
                ("Created", format_date(detail.get("created_at"))),
                ("App version", detail.get("app_version") or "—"),
            ],
            0,
            0,
        )
        self._info_card(
            "Demographics",
            [
                ("Date of birth", detail.get("date_of_birth") or "—"),
                ("Age", f"{detail['age_years']} years" if detail.get("age_years") is not None else "—"),
                ("Height", f"{detail['height_cm']} cm" if detail.get("height_cm") is not None else "—"),
                ("Sex", sex_label(detail.get("sex"))),
            ],
            1,
            0,
        )
        self._info_card(
            "Consent",
            [
                ("18+ confirmed", bool_label(detail.get("consent_18plus"))),
                ("Terms accepted", bool_label(detail.get("consent_terms"))),
            ],
            0,
            1,
        )
        ua = detail.get("user_agent") or "—"
        if len(ua) > 160:
            ua = ua[:160] + "…"
        self._info_card(
            "Technical",
            [
                ("Encryption", detail.get("enc_algo") or "—"),
                ("Front photo path", detail.get("front_photo_path") or "—"),
                ("Side photo path", detail.get("side_photo_path") or "—"),
                ("User agent", ua),
            ],
            1,
            1,
        )
        self._measurements_card(detail.get("measurements") or [])
        self._refresh_scroll_regions()

    def _info_card(self, title: str, rows: list[tuple[str, str]], column: int, row: int) -> None:
        card = ctk.CTkFrame(self.cards_frame, fg_color=PHOTO_BG, corner_radius=CORNER)
        card.grid(row=row, column=column, sticky="nsew", padx=4, pady=6)

        ctk.CTkLabel(
            card,
            text=title.upper(),
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=14, pady=(12, 6))

        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=14, pady=(0, 12))
        for idx, (label, value) in enumerate(rows):
            ctk.CTkLabel(body, text=label, font=ctk.CTkFont(size=12), text_color=MUTED).grid(
                row=idx, column=0, sticky="nw", padx=(0, 12), pady=3
            )
            ctk.CTkLabel(
                body,
                text=value or "—",
                font=ctk.CTkFont(size=12),
                text_color=TEXT,
                wraplength=self._card_value_wrap(),
                justify="left",
                anchor="w",
            ).grid(row=idx, column=1, sticky="nw", pady=3)

    def _measurements_card(self, measurements: list[dict]) -> None:
        card = ctk.CTkFrame(self.cards_frame, fg_color=PHOTO_BG, corner_radius=CORNER)
        card.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=4, pady=6)

        ctk.CTkLabel(
            card,
            text="MEASUREMENTS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=14, pady=(12, 6))

        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkLabel(body, text="Measurement", font=ctk.CTkFont(size=12), text_color=MUTED).grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 4)
        )
        ctk.CTkLabel(body, text="Value", font=ctk.CTkFont(size=12), text_color=MUTED).grid(
            row=0, column=1, sticky="w", pady=(0, 4)
        )
        for idx, m in enumerate(measurements, start=1):
            value = m.get("value")
            unit = m.get("unit", "")
            value_text = f"{value} {unit}".strip() if value is not None else "—"
            ctk.CTkLabel(body, text=m.get("label", m.get("id", "")), font=ctk.CTkFont(size=12), text_color=TEXT).grid(
                row=idx, column=0, sticky="w", padx=(0, 12), pady=2
            )
            ctk.CTkLabel(body, text=value_text, font=ctk.CTkFont(size=12), text_color=TEXT).grid(
                row=idx, column=1, sticky="w", pady=2
            )

    def _clear_cards(self) -> None:
        for child in self.cards_frame.winfo_children():
            child.destroy()

    def _detach_photo_image(self, key: str) -> None:
        """Clear a photo label's image without triggering stale pyimage errors."""
        label = self._photo_labels[key]
        # CTkLabel.configure(image=None) does not clear the underlying tk.Label image.
        label._label.configure(image="")
        if label._image is not None:
            label.configure(image=None)
        self._photo_images.pop(key, None)

    def _set_photo_placeholder(self, key: str, text: str) -> None:
        label = self._photo_labels[key]
        title = self._photo_title_labels[key]
        self._detach_photo_image(key)
        label.configure(text=text)
        title.configure(text=key.capitalize())

    def _show_photo(self, key: str, title: str, data: bytes) -> None:
        self._last_photo_bytes[key] = data
        self._render_photo(key, title, data)

    def _render_photo(self, key: str, title: str, data: bytes) -> None:
        label = self._photo_labels[key]
        title_label = self._photo_title_labels[key]
        try:
            original = Image.open(io.BytesIO(data))
            orig_w, orig_h = original.size
            max_w, max_h = self._photo_max_size()
            fitted = fit_image(original, max_w, max_h)
            self._detach_photo_image(key)
            ctk_image = ctk.CTkImage(light_image=fitted, dark_image=fitted, size=(fitted.width, fitted.height))
            self._photo_images[key] = ctk_image
            label.configure(image=ctk_image, text="")
            title_label.configure(
                text=f"{title}  ·  {orig_w}×{orig_h} px  (preview {fitted.width}×{fitted.height})"
            )
            self._refresh_scroll_regions()
        except Exception as exc:
            self._set_photo_placeholder(key, f"Failed: {exc}")
            title_label.configure(text=title)

    def _card_value_wrap(self) -> int:
        return max((self.detail_scroll.winfo_width() - 120) // 2, 180)

    def _refresh_scroll_regions(self) -> None:
        for frame in (self.list_scroll, self.detail_scroll):
            try:
                canvas = frame._parent_canvas
                frame.update_idletasks()
                bbox = canvas.bbox("all")
                if bbox:
                    canvas.configure(scrollregion=bbox)
            except tk.TclError:
                pass

    def _schedule_wraplengths(self) -> None:
        if self._wrap_job:
            self.after_cancel(self._wrap_job)
        self._wrap_job = self.after(150, self._run_wraplengths)

    def _run_wraplengths(self) -> None:
        self._wrap_job = None
        self._update_wraplengths()

    def _update_wraplengths(self) -> None:
        detail_w = max(self.detail_scroll.winfo_width() - 56, 280)
        self.id_label.configure(wraplength=detail_w)
        list_w = max(self.list_scroll.winfo_width() - 24, 200)
        self.status_label.configure(wraplength=list_w)
        for row in self._list_row_frames:
            for child in row.winfo_children():
                if isinstance(child, ctk.CTkLabel):
                    child.configure(wraplength=list_w)
        self._refresh_scroll_regions()

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        self._schedule_wraplengths()
        if not self._last_photo_bytes:
            return
        size = (event.width, event.height)
        lw, lh = self._last_refit_size
        if abs(size[0] - lw) < 48 and abs(size[1] - lh) < 48:
            return
        self._last_refit_size = size
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(250, self._refit_photos)

    def _refit_photos(self) -> None:
        self._resize_job = None
        if "front" in self._last_photo_bytes:
            self._render_photo("front", "Front", self._last_photo_bytes["front"])
        if "side" in self._last_photo_bytes:
            self._render_photo("side", "Side", self._last_photo_bytes["side"])

    def _delete_selected(self) -> None:
        if not self._selected_id:
            return

        short_id = self._selected_id[:8] + "…"
        if not messagebox.askyesno(
            "Delete submission",
            f"Delete submission {short_id}?\n\nThis removes the database record and encrypted photos from storage.",
        ):
            return

        dialog = ConfirmDeleteDialog(self, self._selected_id)
        self.wait_window(dialog)
        if not dialog.result:
            return

        submission_id = self._selected_id
        self.delete_btn.configure(state="disabled", text="Deleting…")

        def worker() -> None:
            try:
                self.repo.delete_submission(submission_id)
                self.after(0, partial(self._on_delete_done, None))
            except Exception as exc:
                self.after(0, partial(self._on_delete_done, str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_delete_done(self, error: str | None) -> None:
        self.delete_btn.configure(state="normal", text="Delete submission…")
        if error:
            messagebox.showerror("Delete failed", error)
            return

        self._selected_id = None
        self._selected_index = None
        self._last_photo_bytes.clear()
        for key in list(self._photo_images):
            self._detach_photo_image(key)
        self._show_empty_detail()
        self.refresh_list()
        messagebox.showinfo("Deleted", "Submission and photos were permanently deleted.")


def main() -> None:
    try:
        repo = DatasetRepository()
    except RuntimeError as exc:
        root = ctk.CTk()
        root.withdraw()
        messagebox.showerror("Dataset Viewer", str(exc))
        root.destroy()
        return

    if not repo.keys_match_config:
        root = ctk.CTk()
        root.withdraw()
        messagebox.showwarning(
            "Dataset Viewer — key mismatch",
            "Your private key does not match DATASET_PUBLIC_KEY in config.js.\n\n"
            f"Your key derives public:\n{repo._derived_public_b64}\n\n"
            f"Site/config.js public:\n{repo._expected_public_b64}\n\n"
            "Metadata will load, but photo decryption may fail for older submissions.\n\n"
            "Run: python scripts/verify_dataset_keys.py",
        )
        root.destroy()

    app = DatasetViewerApp(repo)
    app.lift()
    app.attributes("-topmost", True)
    app.after(200, lambda: app.attributes("-topmost", False))
    app.mainloop()


def _warn_system_python() -> None:
    if sys.platform != "darwin":
        return
    exe = Path(sys.executable).resolve()
    if "Xcode" in str(exe) or "/usr/bin/python" in str(exe):
        print(
            "Warning: macOS system Python Tk often shows a blank window.\n"
            "If the app looks empty, recreate the venv with Homebrew Python:\n"
            "  brew install python@3.12 python-tk@3.12\n"
            "  rm -rf .venv && /usr/local/bin/python3.12 -m venv .venv\n"
            "  source .venv/bin/activate && pip install -r requirements.txt\n",
            file=sys.stderr,
        )


if __name__ == "__main__":
    _warn_system_python()
    main()
