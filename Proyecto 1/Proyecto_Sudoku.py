import random
import copy
import pandas as pd
import matplotlib.pyplot as plt

class SolucionadorSudokuGenetico:
    """
    Clase que encapsula la lógica para resolver un Sudoku utilizando
    un algoritmo genético híbrido (con búsqueda local).
    Utiliza Pandas para representar internamente el estado del tablero.
    """
    def __init__(self, tablero_base, tam_poblacion=200, max_generaciones=1000):
        # Convertimos la entrada (lista de listas) a DataFrame de Pandas
        self.tablero_base_df = pd.DataFrame(tablero_base)
        self.tam_poblacion = tam_poblacion
        self.max_generaciones = max_generaciones
        # Identificamos las celdas que no deben moverse
        self.celdas_fijas = self._identificar_fijas(self.tablero_base_df)
        
        # Pre-calculamos los índices movibles por fila para no hacerlo en el bucle
        self.indices_movibles = {}
        for f in range(9):
            self.indices_movibles[f] = [c for c in range(9) if (f, c) not in self.celdas_fijas]
            
        self.poblacion = []

    def _identificar_fijas(self, df_tablero):
        """Retorna un conjunto de coordenadas (fila, col) que tienen valor inicial."""
        fijas = set()
        for f in range(9):
            for c in range(9):
                valor = df_tablero.iloc[f, c]
                if valor != 0:
                    fijas.add((f, c))
        return fijas

    def _generar_tablero_valido_filas(self):
        """
        Crea un individuo rellenando los ceros de cada fila con los números
        faltantes (1-9), asegurando que NO haya duplicados en filas.
        Retorna un DataFrame.
        """
        nuevo_df = self.tablero_base_df.copy()
        
        for f in range(9):
            # Obtener valores existentes en la fila
            fila_vals = nuevo_df.iloc[f, :].values
            existentes = {val for val in fila_vals if val != 0}
            
            # Calcular faltantes
            faltantes = list(set(range(1, 10)) - existentes)
            random.shuffle(faltantes)
            
            # Rellenar celdas vacías (0)
            for c in range(9):
                if nuevo_df.iloc[f, c] == 0:
                    nuevo_df.iloc[f, c] = faltantes.pop()
        
        return nuevo_df

    def calcular_penalizacion(self, df_tablero):
        """
        Función de Fitness: Cuenta duplicados en filas, columnas y bloques.
        Objetivo: Minimizar a 0.
        """
        total_errores = 0
        
        # 1. Revisar repeticiones por fila (usando nunique de pandas)
        # axis=1 recorre columnas para cada fila -> cuenta únicos por fila
        unicos_por_fila = df_tablero.nunique(axis=1)
        # Por cada fila, error = 9 - cantidad_unicos
        total_errores += (9 - unicos_por_fila).sum()

        # 2. Revisar repeticiones por columna
        # axis=0 recorre filas para cada columna -> cuenta únicos por columna
        unicos_por_columna = df_tablero.nunique(axis=0)
        total_errores += (9 - unicos_por_columna).sum()
            
        # 3. Revisar subcuadrículas 3x3
        # Iteramos los bloques extrayendo slices del DataFrame
        for f_bloque in range(0, 9, 3):
            for c_bloque in range(0, 9, 3):
                bloque = df_tablero.iloc[f_bloque:f_bloque+3, c_bloque:c_bloque+3].values.flatten()
                total_errores += (9 - len(set(bloque)))
                
        return total_errores

    def _torneo(self, participantes):
        """Selección por torneo: Elige 3 al azar y retorna el mejor."""
        candidatos = random.sample(participantes, 3)
        return min(candidatos, key=self.calcular_penalizacion)

    def _operador_cruce(self, padre_a_df, padre_b_df):
        """Cruce: Copia al padre A e inserta una fila completa del padre B."""
        descendiente = padre_a_df.copy()
        idx_fila = random.randint(0, 8)
        
        # Reemplazar la fila completa usando .iloc
        descendiente.iloc[idx_fila, :] = padre_b_df.iloc[idx_fila, :].values
        return descendiente

    def _operador_mutacion(self, df_tablero, probabilidad=0.1):
        """
        Mutación: Modifica una celda al azar con un valor al azar (1-9),
        respetando las celdas fijas. Se intenta 2 veces.
        """
        for _ in range(2):
            if random.random() < probabilidad:
                f, c = random.randint(0, 8), random.randint(0, 8)
                if (f, c) not in self.celdas_fijas:
                    df_tablero.iloc[f, c] = random.randint(1, 9)

    def _busqueda_local(self, df_tablero, intentos=20):
        """
        Intenta mejorar el individuo intercambiando dos números en la misma fila.
        Si mejora el fitness, conserva el cambio.
        Optimización: Realiza cambios in-place y revierte si no mejora para evitar copias lentas de Pandas.
        """
        # Trabajamos sobre una copia inicial local para no afectar referencias externas inesperadamente,
        # aunque en este flujo genético df_tablero ya suele ser un hijo nuevo.
        # Asumimos que podemos modificar df_tablero (elitismo usa copias previas).
        
        mejor_actual = df_tablero 
        fit_actual = self.calcular_penalizacion(mejor_actual)

        for _ in range(intentos):
            fila_idx = random.randint(0, 8)
            
            # Usamos el pre-cálculo
            movibles = self.indices_movibles[fila_idx]
            
            if len(movibles) < 2:
                continue

            # Selección de columnas
            c1, c2 = random.sample(movibles, 2)
            
            # Guardar valores originales
            val1 = mejor_actual.iloc[fila_idx, c1]
            val2 = mejor_actual.iloc[fila_idx, c2]
            
            # Intercambiar (Swap)
            mejor_actual.iloc[fila_idx, c1] = val2
            mejor_actual.iloc[fila_idx, c2] = val1

            # Calcular nuevo fitness
            fit_vecino = self.calcular_penalizacion(mejor_actual)
            
            # Si mejora (o iguala), nos quedamos con el cambio.
            if fit_vecino < fit_actual:
                fit_actual = fit_vecino
            else:
                # Si NO mejora, REVERTIMOS el cambio (Backtrack)
                mejor_actual.iloc[fila_idx, c1] = val1
                mejor_actual.iloc[fila_idx, c2] = val2
                
        return mejor_actual

    def _inicializar_visualizacion(self):
        plt.ion()
        self.fig, (self.ax_tablero, self.ax_fitness) = plt.subplots(1, 2, figsize=(12, 6))
        self.historial_fitness = []
        self.eje_x = []
        
        # Configurar gráfico de fitness
        self.linea_fitness, = self.ax_fitness.plot([], [], 'b-')
        self.ax_fitness.set_title("Evolución del Fitness (Errores)")
        self.ax_fitness.set_xlabel("Generación")
        self.ax_fitness.set_ylabel("Penalización")
        self.ax_fitness.grid(True)
        
        # Configurar tablero visual
        self.ax_tablero.set_title("Mejor Individuo Actual")
        self.ax_tablero.axis('off')
        
        # Dibujar lineas de la grilla
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5
            # Matplotlib coordenadas 0..1
            self.ax_tablero.axhline(i/9, color='black', linewidth=lw)
            self.ax_tablero.axvline(i/9, color='black', linewidth=lw)
            
        self.textos_tablero = [[None for _ in range(9)] for _ in range(9)]
        # Inicializar textos vacíos
        for f in range(9):
            for c in range(9):
                # Coordenadas: x va de izq a derecha, y de abajo a arriba.
                # Queremos f=0 arriba (y=1), f=8 abajo (y=0).
                x = (c + 0.5) / 9
                y = 1 - (f + 0.5) / 9
                
                # Diferenciar color para celdas fijas
                es_fija = (f, c) in self.celdas_fijas
                color = 'blue' if es_fija else 'black'
                weight = 'bold' if es_fija else 'normal'
                
                self.textos_tablero[f][c] = self.ax_tablero.text(
                    x, y, '', 
                    ha='center', va='center', 
                    fontsize=12, color=color, weight=weight
                )

    def _actualizar_graficos(self, generacion, mejor_individuo_df, score):
        # Actualizar gráfica de fitness
        self.historial_fitness.append(score)
        self.eje_x.append(generacion)
        
        self.linea_fitness.set_data(self.eje_x, self.historial_fitness)
        self.ax_fitness.relim()
        self.ax_fitness.autoscale_view()
        
        # Actualizar números en tablero
        for f in range(9):
            for c in range(9):
                val = mejor_individuo_df.iloc[f, c]
                self.textos_tablero[f][c].set_text(str(val))
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def resolver_problema(self):
        """Ejecuta el ciclo evolutivo principal."""
        print("Generando población inicial...")
        # 1. Población inicial
        self.poblacion = [self._generar_tablero_valido_filas() for _ in range(self.tam_poblacion)]

        self._inicializar_visualizacion()
        
        print("Iniciando evolución...")
        for g in range(self.max_generaciones):
            # Ordenar por fitness (ascendente, menor error es mejor)
            self.poblacion.sort(key=self.calcular_penalizacion)
            
            mejor_individuo = self.poblacion[0]
            score_mejor = self.calcular_penalizacion(mejor_individuo)

            self._actualizar_graficos(g, mejor_individuo, score_mejor)
            
            # Condición de parada exitosa
            if score_mejor == 0:
                print(f"\n[ÉXITO] Solución hallada en la generación {g}")
                return mejor_individuo.values.tolist() # Retornamos lista de listas para compatibilidad
            
            # Elitismo: Conservamos los 'n' mejores
            siguiente_generacion = self.poblacion[:10]
            
            # Completar la nueva generación con hijos
            while len(siguiente_generacion) < self.tam_poblacion:
                padre1 = self._torneo(self.poblacion)
                padre2 = self._torneo(self.poblacion)
                
                hijo = self._operador_cruce(padre1, padre2)
                
                # Aplicar mutación (probabilidad 0.2 según original)
                self._operador_mutacion(hijo, probabilidad=0.2)
                
                # Paso Híbrido: Búsqueda Local (30 intentos según original)
                hijo = self._busqueda_local(hijo, intentos=30)
                
                siguiente_generacion.append(hijo)
            
            self.poblacion = siguiente_generacion
            
            # Imprimir cada 10 generaciones para mayor visibilidad con Pandas
            if g % 10 == 0:
                print(f"Generación {g} | Errores mínimos: {score_mejor}")
                
        print("\n[FIN] Límite de generaciones alcanzado sin solución perfecta.")
        return self.poblacion[0].values.tolist()

def imprimir_tablero(matriz, titulo_cabecera):
    # Si viene un DataFrame, lo convertimos a lista de listas para imprimir igual
    if isinstance(matriz, pd.DataFrame):
        matriz = matriz.values.tolist()
        
    print(f"\n{'='*45}")
    print(f"{titulo_cabecera:^45}")
    print(f"{'='*45}")
    
    encabezados = " ".join([f"C{i+1}" for i in range(9)])
    print(f"     {encabezados}")
    print("   +" + ("---+" * 9))
    
    for i, fila in enumerate(matriz):
        contenido = " | ".join(str(n) for n in fila)
        print(f"F{i+1} | {contenido} |")
        
        # Dibujar líneas divisorias horizontales de bloques 3x3
        if (i + 1) % 3 == 0:
             print("   +" + ("---+" * 9))

if __name__ == "__main__":
    # Definición del problema
    TABLERO_RETO = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    imprimir_tablero(TABLERO_RETO, "SUDOKU INICIAL")
    
    # Instanciar y ejecutar
    motor = SolucionadorSudokuGenetico(TABLERO_RETO)
    mejor_solucion = motor.resolver_problema()
    
    imprimir_tablero(mejor_solucion, "SUDOKU FINAL")
    
    # Recalcular el fitness para mostrarlo al final.
    # Dado que mejor_solucion ya es una lista de listas (no un DF), creamos un DF temporal
    score_final = motor.calcular_penalizacion(pd.DataFrame(mejor_solucion))
    print(f"\n Fitness final (0 es perfecto): {score_final}")

    print("\nVisualización finalizada. Cierre la ventana gráfica para salir.")
    plt.ioff()
    plt.show()