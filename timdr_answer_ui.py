"""Local desktop interface for the deterministic TIMDR answer engine."""
import tkinter as tk
from tkinter import scrolledtext

from timdr_answer_engine import answer


EXAMPLES = (
    "Przykłady: Czy MC K-G jest potwierdzony? | "
    "Czy Weingarten działa bezpośrednio na samej krzywej? | "
    "Czy wolno dopasować PCA GIA na holdoucie? | "
    "Czy AI może samo ogłosić wynik SUPPORTED? | "
    "Dlaczego rho i J nie powinny mieć tego samego źródła?"
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TIMDR Answer Engine — Claim Graph")
        self.geometry("760x500")
        self.minsize(620, 420)
        self.configure(padx=14, pady=14, bg="#111827")

        tk.Label(self, text="TIMDR Answer Engine", fg="#f9fafb", bg="#111827", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(self, text="Deterministyczna odpowiedź ze źródłem; bez Qwen i bez zgadywania.", fg="#cbd5e1", bg="#111827").pack(anchor="w", pady=(2, 12))
        self.question = tk.Entry(self, font=("Segoe UI", 11))
        self.question.pack(fill="x")
        self.question.bind("<Return>", lambda _event: self.run())
        tk.Button(self, text="Sprawdź", command=self.run, bg="#2563eb", fg="white", relief="flat", padx=16, pady=7).pack(anchor="w", pady=10)
        self.status = tk.Label(self, text=EXAMPLES, wraplength=700, justify="left", fg="#93c5fd", bg="#111827")
        self.status.pack(anchor="w", pady=(0, 8))
        self.output = scrolledtext.ScrolledText(self, wrap="word", font=("Consolas", 10), bg="#020617", fg="#e2e8f0", insertbackground="white", relief="flat")
        self.output.pack(fill="both", expand=True)
        self.output.insert("1.0", "Wpisz pytanie dotyczące zakresu Claim Graph.\n")
        self.output.configure(state="disabled")

    def run(self):
        question = self.question.get().strip()
        if not question:
            return
        result = answer(question)
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", result["answer"])
        self.output.configure(state="disabled")
        color = "#86efac" if result["is_in_scope"] else "#fbbf24"
        self.status.configure(text=f"Status: {result['verdict']}", fg=color)


if __name__ == "__main__":
    App().mainloop()
