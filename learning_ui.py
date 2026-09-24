"""Small local UI for the bounded TIMDR source-learning workflow."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext

from environment_policy import EnvironmentPolicyError
from online_learning import OnlineLearningError
from online_ranker import rank_state
from research_queue import build_queue
from skill_guided_learning import run_skill_guided_cycle

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "external_cache"
DEFAULT_CATALOG = ROOT / "online_catalogs.personal_repos.json"


class LearningApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TIMDR Learning UI — źródła i kolejka")
        self.geometry("800x560")
        self.minsize(680, 460)
        self.configure(padx=14, pady=14, bg="#111827")
        self.catalog = tk.StringVar(value=str(DEFAULT_CATALOG))
        self.status = tk.StringVar(value="Gotowe. Nauka źródeł nie tworzy hipotez ani werdyktów TIMDR.")
        tk.Label(self, text="TIMDR Learning UI", fg="#f9fafb", bg="#111827", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(self, text="Pobiera tylko jawne źródła HTTPS i zapisuje lokalny ślad pochodzenia.", fg="#cbd5e1", bg="#111827").pack(anchor="w", pady=(2, 12))
        row = tk.Frame(self, bg="#111827")
        row.pack(fill="x")
        tk.Entry(row, textvariable=self.catalog, font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Wybierz…", command=self.choose_catalog, padx=10).pack(side="left", padx=(8, 0))
        actions = tk.Frame(self, bg="#111827")
        actions.pack(anchor="w", pady=10)
        self.learn_button = tk.Button(actions, text="Pobierz jedną porcję", command=self.start_learning, bg="#2563eb", fg="white", relief="flat", padx=12, pady=7)
        self.learn_button.pack(side="left")
        tk.Button(actions, text="Ranking + kolejka", command=self.refresh_ranking, padx=12, pady=7).pack(side="left", padx=8)
        tk.Button(actions, text="Otwórz raport", command=self.show_report, padx=12, pady=7).pack(side="left")
        tk.Label(self, textvariable=self.status, wraplength=740, justify="left", fg="#93c5fd", bg="#111827").pack(anchor="w", pady=(0, 8))
        self.output = scrolledtext.ScrolledText(self, wrap="word", font=("Consolas", 10), bg="#020617", fg="#e2e8f0", insertbackground="white", relief="flat")
        self.output.pack(fill="both", expand=True)
        self.write("Wybierz katalog i kliknij „Pobierz jedną porcję”.\n")

    def write(self, message: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", message)
        self.output.configure(state="disabled")

    def choose_catalog(self) -> None:
        selected = filedialog.askopenfilename(initialdir=ROOT, filetypes=[("JSON", "*.json")])
        if selected:
            self.catalog.set(selected)

    def start_learning(self) -> None:
        path = Path(self.catalog.get())
        if not path.is_file():
            self.status.set("Nie znaleziono katalogu JSON.")
            return
        self.learn_button.configure(state="disabled")
        self.status.set("Pobieranie trwa w tle…")
        threading.Thread(target=self._learning_worker, args=(path,), daemon=True).start()

    def _learning_worker(self, path: Path) -> None:
        try:
            report = run_skill_guided_cycle(path, CACHE)
            message = (f"Nowe dokumenty: {report['new_documents']}\nPominięte niedostępne: {report['rejected_documents']}\n"
                       f"Karty kandydatów: {report['candidate_cards']}\nHoldout otwarty: {report['holdout_accessed']}\n\nRaport: {report['report_path']}")
            self.after(0, lambda: self._finish_learning("Pobrano porcję źródeł.", message))
        except (OnlineLearningError, EnvironmentPolicyError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.after(0, lambda: self._finish_learning(f"Cykl odrzucony: {exc}", "Brak zmian w Claim Graph ani prerejestracjach."))

    def _finish_learning(self, status: str, message: str) -> None:
        self.status.set(status)
        self.write(message)
        self.learn_button.configure(state="normal")

    def refresh_ranking(self) -> None:
        state_path = CACHE / "online_learning_state.json"
        if not state_path.is_file():
            self.status.set("Najpierw pobierz co najmniej jedną porcję źródeł.")
            return
        ranking = rank_state(state_path)
        ranking_path = CACHE / "online_source_ranking.json"
        ranking_path.write_text(json.dumps(ranking, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        queue = build_queue(ranking_path, state_path)
        queue_path = CACHE / "research_queue.json"
        queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        preview = "\n".join(f"{item['branch']:<18} {item['title']}" for item in queue["entries"][:12])
        self.status.set(f"Odświeżono ranking i kolejkę: {len(queue['entries'])} źródeł.")
        self.write(f"Pierwsze 12 pozycji:\n\n{preview}\n\nKolejka: {queue_path}")

    def show_report(self) -> None:
        report_path = CACHE / "skill_guided_learning_report.json"
        if not report_path.is_file():
            self.status.set("Nie ma jeszcze raportu — uruchom pobieranie.")
            return
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cards = report.get("candidate_cards", [])
        preview = "\n".join(f"{card['title']} → {card['candidate_branches'][0]['branch']}" for card in cards[:20])
        self.status.set(f"Raport: {len(cards)} kart kandydatów.")
        self.write(f"Karty kandydatów (pierwsze 20):\n\n{preview}")


if __name__ == "__main__":
    LearningApp().mainloop()
