import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import main  # Importamos tu logica existente

class ThreadSafeConsole:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget.configure(state='disabled')

    def write(self, msg):
        # Programar la actualización en el hilo principal
        self.text_widget.after(0, self._append, msg)

    def _append(self, msg):
        self.text_widget.configure(state='normal')
        self.text_widget.insert('end', msg)
        self.text_widget.see('end')
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass

class HorarioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Optimización de Horarios - Avanzado")
        self.root.geometry("1400x900")
        
        # Estilos
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- PANEL SUPERIOR: CONFIGURACIÓN ---
        config_frame = ttk.LabelFrame(root, text="Configuración del Algoritmo", padding="10")
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Grid para inputs
        self.vars = {}
        params = [
            ("Tamaño Población", "POPULATION_SIZE", main.POPULATION_SIZE),
            ("Generaciones", "GENERATIONS", main.GENERATIONS),
            ("Elitismo", "ELITISM", main.ELITISM),
            ("Mutación Baja", "MUTATION_LOW", main.MUTATION_LOW),
            ("Mutación Alta", "MUTATION_HIGH", main.MUTATION_HIGH),
            ("Intentos Búsqueda Local", "LOCAL_SEARCH_ATTEMPTS", main.LOCAL_SEARCH_ATTEMPTS),
        ]
        
        for i, (label_text, var_name, default_val) in enumerate(params):
            ttk.Label(config_frame, text=label_text).grid(row=0, column=i*2, padx=5, sticky="e")
            var = tk.DoubleVar(value=default_val) if isinstance(default_val, float) else tk.IntVar(value=default_val)
            entry = ttk.Entry(config_frame, textvariable=var, width=8)
            entry.grid(row=0, column=i*2+1, padx=5, sticky="w")
            self.vars[var_name] = var

        self.btn_run = ttk.Button(config_frame, text="▶ EJECUTAR SIMULACIÓN", command=self.start_optimization)
        self.btn_run.grid(row=0, column=len(params)*2, padx=20, sticky="ew")

        # --- PANEL CENTRAL: PESTAÑAS ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Pestaña 1: Logs (Consola)
        self.tab_logs = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_logs, text="📜 Logs y Progreso")
        
        self.log_area = scrolledtext.ScrolledText(self.tab_logs, state='disabled', font=("Consolas", 10), background="#1e1e1e", foreground="#f0f0f0")
        self.log_area.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña 2: Horarios
        self.tab_horarios = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_horarios, text="📅 Horarios Generados")
        self.notebook_aulas = ttk.Notebook(self.tab_horarios)
        self.notebook_aulas.pack(fill=tk.BOTH, expand=True)

        # Pestaña 3: Datos del Problema
        self.tab_datos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_datos, text="ℹ️ Datos del Problema")
        self.setup_data_tab()

        # Redirección de stdout
        self.original_stdout = sys.stdout
        self.console = ThreadSafeConsole(self.log_area)

    def setup_data_tab(self):
        # Crear sub-pestañas para ver datos
        nb_datos = ttk.Notebook(self.tab_datos)
        nb_datos.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Materias
        f_mat = ttk.Frame(nb_datos)
        nb_datos.add(f_mat, text="Materias")
        self.create_table(f_mat, main.MATERIAS)
        
        # Profesores
        f_prof = ttk.Frame(nb_datos)
        nb_datos.add(f_prof, text="Profesores")
        self.create_table(f_prof, main.PROFESORES)
        
        # Aulas
        f_aula = ttk.Frame(nb_datos)
        nb_datos.add(f_aula, text="Aulas")
        self.create_table(f_aula, main.AULAS)

    def create_table(self, parent, data_list):
        if not data_list: return
        headers = list(data_list[0].__dict__.keys())
        tree = ttk.Treeview(parent, columns=headers, show="headings")
        for h in headers:
            tree.heading(h, text=h.capitalize())
            tree.column(h, width=150)
        
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for item in data_list:
            tree.insert("", "end", values=list(item.__dict__.values()))

    def start_optimization(self):
        # Actualizar parámetros en main
        try:
            main.POPULATION_SIZE = self.vars["POPULATION_SIZE"].get()
            main.GENERATIONS = self.vars["GENERATIONS"].get()
            main.ELITISM = self.vars["ELITISM"].get()
            main.MUTATION_LOW = self.vars["MUTATION_LOW"].get()
            main.MUTATION_HIGH = self.vars["MUTATION_HIGH"].get()
            main.LOCAL_SEARCH_ATTEMPTS = self.vars["LOCAL_SEARCH_ATTEMPTS"].get()
        except ValueError:
            messagebox.showerror("Error", "Por favor revise los valores numéricos de configuración.")
            return

        self.btn_run.config(state="disabled")
        
        # Resetear logs
        self.log_area.configure(state="normal")
        self.log_area.delete(1.0, tk.END)
        self.log_area.configure(state="disabled")
        
        # Limpiar visualización de horarios anterior
        for tab in self.notebook_aulas.tabs():
            self.notebook_aulas.forget(tab)
            
        sys.stdout = self.console
        self.notebook.select(self.tab_logs) # Cambiar a pestaña de logs
        
        # Hilo de ejecución
        threading.Thread(target=self.run_logic, daemon=True).start()

    def run_logic(self):
        try:
            # === REPLICANDO FLUJO EXACTO DE MAIN.PY ===
            print("\n" + "="*80)
            print("SISTEMA DE OPTIMIZACION DE HORARIOS ACADEMICOS".center(80))
            print("Algoritmo Genetico Hibrido (Versión GUI)".center(80))
            print("="*80)
            
            print("\nDatos del problema:")
            print(f"- {len(main.MATERIAS)} materias")
            print(f"- {len(main.PROFESORES)} profesores")
            print(f"- {len(main.AULAS)} aulas")
            print(f"- {main.DIAS} dias x {main.BLOQUES_POR_DIA} bloques = {main.TOTAL_BLOQUES} slots totales")
            
            total_bloques_necesarios = sum(m.horas_semanales for m in main.MATERIAS)
            print(f"- {total_bloques_necesarios} bloques de clase necesarios")
            
            # Generar horario inicial
            print("\n" + "="*80)
            print("GENERANDO HORARIO INICIAL (con conflictos)...")
            print("="*80)
            horario_inicial = main.crear_individuo()
            
            # Llamamos a visualizar_horario de main, que ahora imprime a nuestra consola redirigida
            main.visualizar_horario(horario_inicial, "HORARIO INICIAL (CON CONFLICTOS)")
            
            # Ejecutar algoritmo
            print("\n" + "="*80)
            print("EJECUTANDO ALGORITMO GENETICO...")
            print("="*80 + "\n")
            
            horario_optimizado = main.algoritmo_genetico()
            
            # Visualizar final en consola (logs)
            main.visualizar_horario(horario_optimizado, "HORARIO OPTIMIZADO (SOLUCION)")
            
            print("\nProceso completado!")
            print("="*80)
            
            # Actualizar GUI de horarios (tablas)
            self.root.after(0, lambda: self.show_calendar_results(horario_optimizado))
            
        except Exception as e:
            print(f"\nERROR CRÍTICO DURANTE LA EJECUCIÓN: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.root.after(0, self.restore_ui)

    def restore_ui(self):
        sys.stdout = self.original_stdout
        self.btn_run.config(state="normal")
        messagebox.showinfo("Proceso Terminado", "La optimización ha finalizado con éxito.")

    def show_calendar_results(self, horario):
        # Crear vista de calendario como en la versión anterior
        dias_nombres = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
        bloques_nombres = ["8:00-9:30", "9:30-11:00", "11:00-12:30", 
                           "12:30-14:00", "14:00-15:30", "15:30-17:00"]
        
        style = ttk.Style()
        style.configure("Treeview", rowheight=40)
        
        for aula in main.AULAS:
            frame_aula = ttk.Frame(self.notebook_aulas)
            self.notebook_aulas.add(frame_aula, text=aula.nombre)
            
            columns = ("Bloque",) + tuple(dias_nombres)
            tree = ttk.Treeview(frame_aula, columns=columns, show="headings")
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=150, anchor="center")
            tree.column("Bloque", width=100, anchor="center")
            
            tree.pack(fill=tk.BOTH, expand=True)
            
            for bloque_idx in range(main.BLOQUES_POR_DIA):
                row_values = [bloques_nombres[bloque_idx]]
                for dia_idx in range(main.DIAS):
                    bloques_en_slot = [b for b in horario 
                                      if b.aula_id == aula.id 
                                      and b.dia == dia_idx 
                                      and b.bloque == bloque_idx]
                    
                    texto = ""
                    if bloques_en_slot:
                        m = next((m for m in main.MATERIAS if m.id == bloques_en_slot[0].materia_id), None)
                        p = next((p for p in main.PROFESORES if p.id == bloques_en_slot[0].profesor_id), None)
                        if m and p:
                            texto = f"{m.nombre}\n{p.nombre}"
                            if len(bloques_en_slot) > 1: texto += " ⚠️"
                    row_values.append(texto)
                tree.insert("", "end", values=row_values)
        
        # No cambiamos el foco automaticamente para que el usuario pueda seguir viendo los logs si quiere
        # self.notebook.select(self.tab_horarios)

if __name__ == "__main__":
    root = tk.Tk()
    app = HorarioApp(root)
    root.mainloop()
