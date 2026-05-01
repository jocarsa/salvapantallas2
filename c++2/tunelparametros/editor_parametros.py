import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os


PARAM_FILE = "params_tunel.txt"
EXECUTABLE = "./tunel_cuda"


PARAMS = {
    "width": 1920,
    "height": 1080,
    "fps": 60,
    "duracion": 3600,

    "num_lados": 64,
    "max_capas": 260,

    "velocidad_expansion": 1.045,
    "radio_inicial": 8.0,

    "noise_strength": 0.22,
    "noise_scale": 1.7,
    "noise_octaves": 4,

    "light_radius": 460.0,
    "light_falloff_power": 2.6,
    "light_intensity": 360.0,
    "specular_intensity": 280.0,

    "light_angle_speed": 0.014,
    "light_radial_base": 0.56,
    "light_radial_amp": 0.18,
    "light_depth_base": 0.50,
    "light_depth_amp": 0.32,
    "light_radial_freq": 0.65,
    "light_depth_freq": 0.27,

    "tunnel_rotation_speed": 0.012,
    "movement_randomness": 1.2,
    "movement_damping": 0.97,

    "margin_x": 420.0,
    "margin_y": 260.0,
    "return_force": 1.8,

    "preview_frame": 100,
}


GROUPS = {
    "Vídeo": [
        "width", "height", "fps", "duracion"
    ],
    "Geometría": [
        "num_lados", "max_capas",
        "velocidad_expansion", "radio_inicial"
    ],
    "Ruido de superficie": [
        "noise_strength", "noise_scale", "noise_octaves"
    ],
    "Luz": [
        "light_radius", "light_falloff_power",
        "light_intensity", "specular_intensity",
        "light_angle_speed",
        "light_radial_base", "light_radial_amp",
        "light_depth_base", "light_depth_amp",
        "light_radial_freq", "light_depth_freq"
    ],
    "Movimiento": [
        "tunnel_rotation_speed",
        "movement_randomness",
        "movement_damping",
        "margin_x", "margin_y", "return_force"
    ],
    "Preview": [
        "preview_frame"
    ]
}


class EditorParametros:
    def __init__(self, root):
        self.root = root
        self.root.title("jocarsa | editor túnel CUDA")
        self.root.geometry("620x760")

        self.vars = {}

        self.crear_ui()
        self.cargar_si_existe()

    def crear_ui(self):
        contenedor = ttk.Frame(self.root, padding=14)
        contenedor.pack(fill="both", expand=True)

        titulo = ttk.Label(
            contenedor,
            text="Editor de parámetros del túnel CUDA",
            font=("Ubuntu", 16, "bold")
        )
        titulo.pack(anchor="w", pady=(0, 12))

        canvas = tk.Canvas(contenedor)
        scroll = ttk.Scrollbar(
            contenedor,
            orient="vertical",
            command=canvas.yview
        )

        self.form = ttk.Frame(canvas)

        self.form.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window(
            (0, 0),
            window=self.form,
            anchor="nw"
        )

        canvas.configure(yscrollcommand=scroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for grupo, claves in GROUPS.items():
            frame = ttk.LabelFrame(
                self.form,
                text=grupo,
                padding=10
            )
            frame.pack(fill="x", pady=8)

            for clave in claves:
                fila = ttk.Frame(frame)
                fila.pack(fill="x", pady=3)

                label = ttk.Label(
                    fila,
                    text=clave,
                    width=28
                )
                label.pack(side="left")

                var = tk.StringVar(
                    value=str(PARAMS[clave])
                )

                entrada = ttk.Entry(
                    fila,
                    textvariable=var
                )
                entrada.pack(side="left", fill="x", expand=True)

                self.vars[clave] = var

        botones = ttk.Frame(contenedor)
        botones.pack(fill="x", pady=12)

        ttk.Button(
            botones,
            text="Guardar TXT",
            command=self.guardar
        ).pack(side="left", padx=4)

        ttk.Button(
            botones,
            text="Preview frame",
            command=self.preview
        ).pack(side="left", padx=4)

        ttk.Button(
            botones,
            text="Render vídeo completo",
            command=self.render
        ).pack(side="left", padx=4)

    def cargar_si_existe(self):
        if not os.path.exists(PARAM_FILE):
            return

        with open(PARAM_FILE, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()

                if not linea or linea.startswith("#"):
                    continue

                if "=" not in linea:
                    continue

                clave, valor = linea.split("=", 1)
                clave = clave.strip()
                valor = valor.strip()

                if clave in self.vars:
                    self.vars[clave].set(valor)

    def guardar(self):
        try:
            with open(PARAM_FILE, "w", encoding="utf-8") as f:
                for clave in PARAMS:
                    valor = self.vars[clave].get().strip()
                    f.write(f"{clave}={valor}\n")

            messagebox.showinfo(
                "Guardado",
                f"Archivo guardado: {PARAM_FILE}"
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                str(e)
            )

    def preview(self):
        self.guardar()

        try:
            subprocess.run(
                [
                    EXECUTABLE,
                    "--params",
                    PARAM_FILE,
                    "--preview"
                ],
                check=True
            )

        except Exception as e:
            messagebox.showerror(
                "Error ejecutando preview",
                str(e)
            )

    def render(self):
        self.guardar()

        try:
            subprocess.Popen(
                [
                    EXECUTABLE,
                    "--params",
                    PARAM_FILE
                ]
            )

        except Exception as e:
            messagebox.showerror(
                "Error ejecutando render",
                str(e)
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = EditorParametros(root)
    root.mainloop()
