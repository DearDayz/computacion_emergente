import random
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from collections import defaultdict

# DEFINICION DE ESTRUCTURAS DE DATOS

@dataclass
class Materia:
    id: int
    nombre: str
    profesor_id: int
    horas_semanales: int  # cantidad de bloques necesarios

@dataclass
class Profesor:
    id: int
    nombre: str
    preferencias_horarias: List[Tuple[int, int]]  # [(dia, bloque), ...]
    preferencias_difusas: Dict[Tuple[int, int], float] = field(default_factory=dict)  # opcional: (dia, bloque) -> grado [0.0-1.0]

@dataclass
class Aula:
    id: int
    nombre: str
    capacidad: int

@dataclass
class Bloque:
    materia_id: int
    profesor_id: int
    aula_id: int
    dia: int  # 0=Lunes, 1=Martes, ..., 4=Viernes
    bloque: int  # 0-5 (6 bloques por dia)

# CONFIGURACION DEL PROBLEMA

DIAS = 5  # Lunes a Viernes
BLOQUES_POR_DIA = 6  # 6 bloques horarios por dia
TOTAL_BLOQUES = DIAS * BLOQUES_POR_DIA

# Parametros del algoritmo (ajustables)
POPULATION_SIZE = 300
GENERATIONS = 2000
ELITISM = 20
MUTATION_LOW = 0.15
MUTATION_HIGH = 0.3
LOCAL_SEARCH_ATTEMPTS = 50

# Opcional: activar uso de preferencias difusas (0.0 a 1.0)
USE_FUZZY_PREFERENCES = True

# Datos de ejemplo
MATERIAS = [
    Materia(1, "Algoritmos", 1, 4),
    Materia(2, "Bases de Datos", 2, 3),
    Materia(3, "Redes", 3, 3),
    Materia(4, "Inteligencia Artificial", 1, 3),
    Materia(5, "Programacion Web", 2, 4),
]

PROFESORES = [
    Profesor(1, "Prof. Garcia", [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2), (2, 3), (4, 0)]),  # prefiere lunes, miercoles, viernes
    Profesor(2, "Prof. Martinez", [(1, 0), (1, 1), (1, 2), (1, 3), (3, 0), (3, 1), (3, 2), (3, 3)]),  # prefiere martes y jueves
    Profesor(3, "Prof. Lopez", [(0, 2), (0, 3), (2, 3), (2, 4), (4, 1), (4, 2), (4, 3), (4, 4)]),  # prefiere lunes, miercoles, viernes
]

AULAS = [
    Aula(1, "Aula A", 30),
    Aula(2, "Aula B", 25),
    Aula(3, "Aula C", 35),
]

# REPRESENTACION DEL INDIVIDUO

# Preferencias difusas de ejemplo: construir mapa para todos los slots
def construir_preferencias_difusas():
    for profesor in PROFESORES:
        dias_preferidos = {dia for (dia, _bloque) in profesor.preferencias_horarias}
        mapa: Dict[Tuple[int, int], float] = {}
        for d in range(DIAS):
            for b in range(BLOQUES_POR_DIA):
                if (d, b) in profesor.preferencias_horarias:
                    grado = 1.0
                elif d in dias_preferidos:
                    grado = 0.7
                else:
                    grado = 0.3
                mapa[(d, b)] = grado
        profesor.preferencias_difusas = mapa

if USE_FUZZY_PREFERENCES:
    construir_preferencias_difusas()

def crear_individuo() -> List[Bloque]:
    """Crea un horario aleatorio respetando cantidad de horas por materia"""
    horario = []
    
    for materia in MATERIAS:
        for _ in range(materia.horas_semanales):
            bloque = Bloque(
                materia_id=materia.id,
                profesor_id=materia.profesor_id,
                aula_id=random.choice(AULAS).id,
                dia=random.randint(0, DIAS - 1),
                bloque=random.randint(0, BLOQUES_POR_DIA - 1)
            )
            horario.append(bloque)
    
    return horario

# FUNCION FITNESS

def fitness(individuo: List[Bloque]) -> int:
    """Calcula fitness basado en conflictos. Menor es mejor (0 = perfecto)"""
    penalizacion = 0
    
    # RESTRICCIONES DURAS (peso alto)
    
    # 1. Conflicto de aulas: un aula no puede tener dos clases simultaneas
    aulas_ocupadas = defaultdict(list)
    for bloque in individuo:
        clave = (bloque.aula_id, bloque.dia, bloque.bloque)
        aulas_ocupadas[clave].append(bloque)
    
    for clave, bloques in aulas_ocupadas.items():
        if len(bloques) > 1:
            penalizacion += 100 * (len(bloques) - 1)
    
    # 2. Conflicto de profesores: un profesor no puede dictar dos clases simultaneas
    profesores_ocupados = defaultdict(list)
    for bloque in individuo:
        clave = (bloque.profesor_id, bloque.dia, bloque.bloque)
        profesores_ocupados[clave].append(bloque)
    
    for clave, bloques in profesores_ocupados.items():
        if len(bloques) > 1:
            penalizacion += 100 * (len(bloques) - 1)
    
    # RESTRICCIONES BLANDAS (peso menor)
    
    # 3. Preferencias horarias de profesores (con soporte opcional difuso)
    for bloque in individuo:
        profesor = next(p for p in PROFESORES if p.id == bloque.profesor_id)
        if USE_FUZZY_PREFERENCES and profesor.preferencias_difusas:
            grado = profesor.preferencias_difusas.get((bloque.dia, bloque.bloque), 0.0)
            penalizacion += int(round(5 * (1.0 - grado)))
        else:
            if (bloque.dia, bloque.bloque) not in profesor.preferencias_horarias:
                penalizacion += 5
    
    # 4. Distribucion equilibrada: evitar muchas clases el mismo dia
    materias_por_dia = defaultdict(lambda: defaultdict(int))
    for bloque in individuo:
        materias_por_dia[bloque.materia_id][bloque.dia] += 1
    
    for materia_id, dias_dict in materias_por_dia.items():
        for dia, cantidad in dias_dict.items():
            if cantidad > 2:  # mas de 2 bloques de una materia en un dia
                penalizacion += 10 * (cantidad - 2)
    
    # 5. Evitar bloques consecutivos muy largos de la misma materia
    for dia in range(DIAS):
        bloques_dia = [b for b in individuo if b.dia == dia]
        bloques_dia.sort(key=lambda x: x.bloque)
        
        consecutivos = 1
        for i in range(1, len(bloques_dia)):
            if (bloques_dia[i].materia_id == bloques_dia[i-1].materia_id and 
                bloques_dia[i].bloque == bloques_dia[i-1].bloque + 1):
                consecutivos += 1
                if consecutivos > 2:  # mas de 2 bloques consecutivos
                    penalizacion += 3
            else:
                consecutivos = 1
    
    return penalizacion

# OPERADORES GENETICOS

def seleccion(poblacion: List[List[Bloque]]) -> List[Bloque]:
    """Seleccion por torneo de 3 individuos"""
    torneo = random.sample(poblacion, 3)
    torneo.sort(key=lambda x: fitness(x))
    return torneo[0]

def cruzar(padre1: List[Bloque], padre2: List[Bloque]) -> List[Bloque]:
    """Cruce uniforme conservando cantidad de bloques por materia"""
    hijo = []
    
    # Agrupar bloques por materia en ambos padres
    for materia in MATERIAS:
        bloques_p1 = [b for b in padre1 if b.materia_id == materia.id]
        bloques_p2 = [b for b in padre2 if b.materia_id == materia.id]
        
        # Tomar bloques de ambos padres alternadamente
        for i in range(materia.horas_semanales):
            if i < len(bloques_p1) and i < len(bloques_p2):
                if random.random() < 0.5:
                    hijo.append(bloques_p1[i])
                else:
                    hijo.append(bloques_p2[i])
            elif i < len(bloques_p1):
                hijo.append(bloques_p1[i])
            elif i < len(bloques_p2):
                hijo.append(bloques_p2[i])
            else:
                # Crear nuevo bloque si falta
                nuevo_bloque = Bloque(
                    materia_id=materia.id,
                    profesor_id=materia.profesor_id,
                    aula_id=random.choice(AULAS).id,
                    dia=random.randint(0, DIAS - 1),
                    bloque=random.randint(0, BLOQUES_POR_DIA - 1)
                )
                hijo.append(nuevo_bloque)
    
    return hijo

def mutar(individuo: List[Bloque], prob: float = 0.2):
    """Mutacion: cambiar aleatoriamente dia, bloque o aula"""
    for bloque in individuo:
        if random.random() < prob:
            tipo_mutacion = random.randint(0, 2)
            if tipo_mutacion == 0:
                bloque.dia = random.randint(0, DIAS - 1)
            elif tipo_mutacion == 1:
                bloque.bloque = random.randint(0, BLOQUES_POR_DIA - 1)
            else:
                bloque.aula_id = random.choice(AULAS).id

# BUSQUEDA LOCAL HIBRIDA

def busqueda_local(individuo: List[Bloque], intentos: int = 50) -> List[Bloque]:
    """Mejora local inteligente resolviendo conflictos especificos"""
    from copy import deepcopy
    mejor = [deepcopy(b) for b in individuo]
    mejor_fitness = fitness(mejor)
    
    for _ in range(intentos):
        if mejor_fitness == 0:
            break
        
        candidato = [deepcopy(b) for b in mejor]
        
        # Detectar bloques conflictivos
        aulas_ocupadas = defaultdict(list)
        profesores_ocupados = defaultdict(list)
        
        for i, bloque in enumerate(candidato):
            aulas_ocupadas[(bloque.aula_id, bloque.dia, bloque.bloque)].append(i)
            profesores_ocupados[(bloque.profesor_id, bloque.dia, bloque.bloque)].append(i)
        
        # Identificar indices con conflictos
        indices_conflicto = set()
        for indices in aulas_ocupadas.values():
            if len(indices) > 1:
                indices_conflicto.update(indices[1:])
        for indices in profesores_ocupados.values():
            if len(indices) > 1:
                indices_conflicto.update(indices[1:])
        
        # Si hay conflictos, resolver uno
        if indices_conflicto:
            idx = random.choice(list(indices_conflicto))
        else:
            # Si no hay conflictos duros, mejorar restricciones blandas
            idx = random.randint(0, len(candidato) - 1)
        
        # Buscar una posicion valida
        for _ in range(20):
            dia_nuevo = random.randint(0, DIAS - 1)
            bloque_nuevo = random.randint(0, BLOQUES_POR_DIA - 1)
            aula_nueva = random.choice(AULAS).id
            
            # Verificar si la nueva posicion esta libre
            ocupado = False
            for i, b in enumerate(candidato):
                if i != idx:
                    if (b.aula_id == aula_nueva and b.dia == dia_nuevo and b.bloque == bloque_nuevo):
                        ocupado = True
                        break
                    if (b.profesor_id == candidato[idx].profesor_id and b.dia == dia_nuevo and b.bloque == bloque_nuevo):
                        ocupado = True
                        break
            
            if not ocupado:
                candidato[idx].dia = dia_nuevo
                candidato[idx].bloque = bloque_nuevo
                candidato[idx].aula_id = aula_nueva
                break
        else:
            # Si no encuentra posicion libre, cambiar aleatoriamente
            candidato[idx].dia = random.randint(0, DIAS - 1)
            candidato[idx].bloque = random.randint(0, BLOQUES_POR_DIA - 1)
            candidato[idx].aula_id = random.choice(AULAS).id
        
        nuevo_fitness = fitness(candidato)
        if nuevo_fitness < mejor_fitness:
            mejor = candidato
            mejor_fitness = nuevo_fitness
    
    return mejor

# ALGORITMO GENETICO PRINCIPAL

def algoritmo_genetico():
    """Algoritmo genetico hibrido para optimizacion de horarios"""
    poblacion = [crear_individuo() for _ in range(POPULATION_SIZE)]

    for generacion in range(GENERATIONS):
        poblacion.sort(key=lambda x: fitness(x))
        mejor = poblacion[0]
        fit = fitness(mejor)

        if fit == 0:
            print(f"\nSolucion encontrada en generacion {generacion}")
            return mejor

        nueva_poblacion = poblacion[:ELITISM]  # elitismo

        while len(nueva_poblacion) < POPULATION_SIZE:
            p1 = seleccion(poblacion)
            p2 = seleccion(poblacion)

            hijo = cruzar(p1, p2)
            
            # Mutacion adaptativa: mas probabilidad si fitness alto
            prob_mutacion = MUTATION_LOW if fit < 50 else MUTATION_HIGH
            mutar(hijo, prob=prob_mutacion)

            # Busqueda local hibrida intensiva
            hijo = busqueda_local(hijo, intentos=LOCAL_SEARCH_ATTEMPTS)

            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

        if generacion % 100 == 0:
            print(f"Gen {generacion} | Mejor fitness: {fit}")

    print("\nNo se encontro solucion perfecta")
    return poblacion[0]

# VISUALIZACION

def visualizar_horario(horario: List[Bloque], titulo: str = "HORARIO"):
    """Muestra el horario en formato tabla"""
    print("\n" + "="*80)
    print(f"{titulo.center(80)}")
    print("="*80)
    print(f"Fitness: {fitness(horario)}\n")
    
    dias_nombres = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes"]
    bloques_nombres = ["8:00-9:30", "9:30-11:00", "11:00-12:30", 
                       "12:30-14:00", "14:00-15:30", "15:30-17:00"]
    
    # Crear matriz del horario
    for aula in AULAS:
        print(f"\n{aula.nombre}:")
        print("-" * 80)
        print(f"{'Bloque':<15}", end="")
        for dia in dias_nombres:
            print(f"{dia:<13}", end="")
        print()
        print("-" * 80)
        
        for bloque_idx in range(BLOQUES_POR_DIA):
            print(f"{bloques_nombres[bloque_idx]:<15}", end="")
            
            for dia_idx in range(DIAS):
                bloques_en_slot = [b for b in horario 
                                  if b.aula_id == aula.id 
                                  and b.dia == dia_idx 
                                  and b.bloque == bloque_idx]
                
                if bloques_en_slot:
                    materia = next(m for m in MATERIAS if m.id == bloques_en_slot[0].materia_id)
                    texto = materia.nombre[:10]
                    if len(bloques_en_slot) > 1:
                        texto += "(!)"  # indica conflicto
                    print(f"{texto:<13}", end="")
                else:
                    print(f"{'---':<13}", end="")
            print()
    
    # Mostrar conflictos
    print("\n" + "-"*80)
    print("ANALISIS DE CONFLICTOS:")
    print("-"*80)
    
    # Conflictos de aulas
    aulas_ocupadas = defaultdict(list)
    for bloque in horario:
        clave = (bloque.aula_id, bloque.dia, bloque.bloque)
        aulas_ocupadas[clave].append(bloque)
    
    conflictos_aulas = sum(1 for bloques in aulas_ocupadas.values() if len(bloques) > 1)
    print(f"Conflictos de aulas: {conflictos_aulas}")
    
    # Conflictos de profesores
    profesores_ocupados = defaultdict(list)
    for bloque in horario:
        clave = (bloque.profesor_id, bloque.dia, bloque.bloque)
        profesores_ocupados[clave].append(bloque)
    
    conflictos_profesores = sum(1 for bloques in profesores_ocupados.values() if len(bloques) > 1)
    print(f"Conflictos de profesores: {conflictos_profesores}")
    
    # Violaciones de preferencias
    violaciones_pref = 0
    for bloque in horario:
        profesor = next(p for p in PROFESORES if p.id == bloque.profesor_id)
        if (bloque.dia, bloque.bloque) not in profesor.preferencias_horarias:
            violaciones_pref += 1
    print(f"Violaciones de preferencias: {violaciones_pref}")
    
    print("="*80 + "\n")

# PROGRAMA PRINCIPAL

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SISTEMA DE OPTIMIZACION DE HORARIOS ACADEMICOS".center(80))
    print("Algoritmo Genetico Hibrido".center(80))
    print("="*80)
    
    print("\nDatos del problema:")
    print(f"- {len(MATERIAS)} materias")
    print(f"- {len(PROFESORES)} profesores")
    print(f"- {len(AULAS)} aulas")
    print(f"- {DIAS} dias x {BLOQUES_POR_DIA} bloques = {TOTAL_BLOQUES} slots totales")
    
    total_bloques_necesarios = sum(m.horas_semanales for m in MATERIAS)
    print(f"- {total_bloques_necesarios} bloques de clase necesarios")
    
    # Generar horario inicial con conflictos
    print("\n" + "="*80)
    print("GENERANDO HORARIO INICIAL (con conflictos)...")
    print("="*80)
    horario_inicial = crear_individuo()
    visualizar_horario(horario_inicial, "HORARIO INICIAL (CON CONFLICTOS)")
    
    # Ejecutar algoritmo genetico
    print("\n" + "="*80)
    print("EJECUTANDO ALGORITMO GENETICO...")
    print("="*80 + "\n")
    
    horario_optimizado = algoritmo_genetico()
    
    # Mostrar resultado
    visualizar_horario(horario_optimizado, "HORARIO OPTIMIZADO (SOLUCION)")
    
    print("\nProceso completado!")
    print("="*80)
