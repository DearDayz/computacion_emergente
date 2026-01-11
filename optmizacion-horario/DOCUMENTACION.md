# SISTEMA DE OPTIMIZACIÓN DE HORARIOS ACADÉMICOS
## Algoritmos Genéticos Aplicados

---

## 1. INTRODUCCIÓN

El problema de asignación de horarios académicos es un problema de optimización combinatoria NP-completo. Consiste en asignar materias, profesores y aulas a bloques horarios específicos, respetando múltiples restricciones que pueden ser duras (obligatorias) o blandas (preferibles).

### Objetivo del Sistema
Diseñar horarios académicos que:
- Eviten solapamientos de aulas y profesores
- Respeten preferencias horarias
- Distribuyan equilibradamente las materias
- Minimicen conflictos en general

---

## 2. ALGORITMOS GENÉTICOS - EXPLICACIÓN

### 2.1 ¿Qué son los Algoritmos Genéticos?

Los Algoritmos Genéticos (AG) son técnicas de búsqueda y optimización inspiradas en la evolución biológica natural. Utilizan conceptos como:
- **Selección natural**: Supervivencia del más apto
- **Reproducción**: Combinación de soluciones
- **Mutación**: Variación aleatoria
- **Evolución**: Mejora generacional

### 2.2 Componentes Principales

#### a) Población
Conjunto de individuos (posibles soluciones). En nuestro caso, cada individuo es un horario completo.

#### b) Individuo (Cromosoma)
Representación de una solución al problema. En este sistema:
- Un individuo = Lista de bloques horarios
- Cada bloque contiene: materia, profesor, aula, día y hora

#### c) Función Fitness
Evalúa la calidad de cada individuo. Retorna un valor numérico donde:
- **Fitness = 0**: Solución perfecta (sin conflictos)
- **Fitness > 0**: Cantidad de penalizaciones acumuladas

#### d) Selección
Proceso de elegir individuos para reproducción. Utilizamos **selección por torneo**:
- Se eligen 3 individuos al azar
- Se selecciona el mejor de los 3
- Favorece buenos individuos sin eliminar diversidad

#### e) Cruce (Crossover)
Combina dos padres para crear un hijo. En nuestro sistema:
- **Cruce de un punto**: Se divide cada padre en un punto aleatorio
- Se toma la primera parte del padre 1 y la segunda del padre 2
- Se ajusta para mantener cantidad correcta de bloques por materia

#### f) Mutación
Introduce variación aleatoria. Cambios aplicados:
- Cambio de día
- Cambio de bloque horario
- Cambio de aula

#### g) Elitismo
Preserva los mejores individuos de cada generación. En este sistema:
- Los 10 mejores pasan directamente a la siguiente generación
- Garantiza que no se pierdan buenas soluciones

### 2.3 Algoritmo Genético Híbrido

Este sistema implementa un **AG Híbrido** que combina:
1. **Operadores genéticos clásicos** (selección, cruce, mutación)
2. **Búsqueda local** para refinamiento de soluciones

La búsqueda local intenta mejorar cada hijo generado mediante cambios locales iterativos, lo que acelera la convergencia hacia soluciones óptimas.

### 2.4 Flujo del Algoritmo

```
1. Generar población inicial aleatoria (200 individuos)
2. MIENTRAS no se encuentre solución perfecta Y generaciones < 1000:
   a. Evaluar fitness de todos los individuos
   b. Ordenar población por fitness (mejor primero)
   c. Si fitness del mejor = 0: SOLUCIÓN ENCONTRADA
   d. Crear nueva población:
      - Mantener 10 mejores (elitismo)
      - Hasta completar 200 individuos:
        * Seleccionar dos padres por torneo
        * Cruzar padres para crear hijo
        * Mutar hijo con probabilidad 20%
        * Aplicar búsqueda local al hijo
        * Agregar hijo a nueva población
   e. Reemplazar población antigua con nueva
   f. Mostrar progreso cada 50 generaciones
3. Retornar mejor solución encontrada
```

---

## 3. MODELACIÓN DEL PROBLEMA

### 3.1 Estructuras de Datos

#### Materia
```python
- id: identificador único
- nombre: nombre de la materia
- profesor_id: profesor asignado
- horas_semanales: bloques necesarios por semana
```

#### Profesor
```python
- id: identificador único
- nombre: nombre del profesor
- preferencias_horarias: lista de (día, bloque) preferidos
```

#### Aula
```python
- id: identificador único
- nombre: nombre del aula
- capacidad: cantidad de estudiantes
```

#### Bloque
```python
- materia_id: materia asignada
- profesor_id: profesor que dicta
- aula_id: aula donde se dicta
- dia: día de la semana (0-4)
- bloque: bloque horario (0-5)
```

### 3.2 Representación del Individuo

Un horario completo se representa como una lista de bloques. Cada materia tiene la cantidad de bloques según sus horas semanales.

**Ejemplo**: Si "Algoritmos" requiere 4 horas semanales, tendrá 4 bloques en el horario.

---

## 4. FUNCIÓN FITNESS Y RESTRICCIONES

### 4.1 Restricciones Duras (Penalización: 100 puntos c/u)

Estas restricciones NO pueden violarse en una solución válida:

#### R1: Conflicto de Aulas
**Descripción**: Un aula no puede tener dos clases simultáneamente.

**Implementación**:
```python
- Crear diccionario: (aula, día, bloque) → lista de bloques
- Si un slot tiene más de 1 bloque: penalización = 100 * (n-1)
```

**Ejemplo de conflicto**:
- Lunes 8:00, Aula A: "Algoritmos" Y "Bases de Datos" → CONFLICTO

#### R2: Conflicto de Profesores
**Descripción**: Un profesor no puede dictar dos clases simultáneamente.

**Implementación**:
```python
- Crear diccionario: (profesor, día, bloque) → lista de bloques
- Si un profesor está en más de 1 lugar: penalización = 100 * (n-1)
```

**Ejemplo de conflicto**:
- Martes 10:00: Prof. García en Aula A y Aula B → CONFLICTO

### 4.2 Restricciones Blandas (Penalizaciones menores)

Estas restricciones son deseables pero no estrictamente obligatorias:

#### R3: Preferencias Horarias (Penalización: 5 puntos c/u)
**Descripción**: Respetar preferencias horarias de profesores.

**Implementación**:
```python
- Por cada bloque fuera de horarios preferidos: +5 puntos
```

**Ejemplo**:
- Prof. García prefiere lunes y miércoles
- Si se le asigna clase el viernes: +5 de penalización

#### R4: Distribución Equilibrada (Penalización: 10 puntos c/u)
**Descripción**: Evitar concentrar muchas clases de una materia en un mismo día.

**Implementación**:
```python
- Si una materia tiene más de 2 bloques en un día: +10 * (n-2)
```

**Ejemplo**:
- "Programación Web" tiene 4 bloques el lunes → +20 puntos

#### R5: Bloques Consecutivos (Penalización: 3 puntos c/u)
**Descripción**: Evitar más de 2 bloques consecutivos de la misma materia.

**Implementación**:
```python
- Si hay más de 2 bloques seguidos: +3 por cada exceso
```

**Ejemplo**:
- Lunes: "Algoritmos" en bloques 1, 2 y 3 → +3 puntos

### 4.3 Cálculo Total del Fitness

```
Fitness Total = Σ(Penalizaciones de R1) + 
                Σ(Penalizaciones de R2) +
                Σ(Penalizaciones de R3) +
                Σ(Penalizaciones de R4) +
                Σ(Penalizaciones de R5)

Objetivo: Minimizar Fitness → Ideal = 0
```

---

## 5. OPERADORES GENÉTICOS DETALLADOS

### 5.1 Selección por Torneo
```python
def seleccion(poblacion):
    1. Elegir 3 individuos al azar
    2. Evaluar fitness de los 3
    3. Retornar el de mejor fitness (menor valor)
```

**Ventajas**:
- Rápida de computar
- Mantiene diversidad
- Presión selectiva ajustable

### 5.2 Cruce de Un Punto con Ajuste
```python
def cruzar(padre1, padre2):
    1. Elegir punto de corte aleatorio
    2. hijo = padre1[:punto] + padre2[punto:]
    3. Ajustar hijo para mantener horas correctas por materia
    4. Si faltan bloques, agregar nuevos aleatorios
    5. Si sobran bloques, eliminar excedentes
```

**Ejemplo**:
```
Padre1: [A1, A2, B1, B2, C1]
Padre2: [A3, B3, B4, C2, C3]
Punto: 2

Hijo sin ajustar: [A1, A2, B4, C2, C3]
Hijo ajustado: [A1, A2, B1, B2, C1] (con ajustes)
```

### 5.3 Mutación
```python
def mutar(individuo, prob=0.2):
    Para cada bloque:
        Si random() < prob:
            Tipo = aleatorio(0, 2)
            Si tipo == 0: cambiar día
            Si tipo == 1: cambiar bloque horario
            Si tipo == 2: cambiar aula
```

**Probabilidad**: 20% por bloque
**Efecto**: Introduce diversidad y explora nuevas soluciones

### 5.4 Búsqueda Local
```python
def busqueda_local(individuo, intentos=30):
    mejor = individuo
    Para i en 1..intentos:
        candidato = mejor (copia)
        Elegir bloque aleatorio
        Cambiar su posición (día y hora)
        Si fitness(candidato) < fitness(mejor):
            mejor = candidato
    Retornar mejor
```

**Objetivo**: Refinamiento local de soluciones prometedoras

---

## 6. PARÁMETROS DEL SISTEMA

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Tamaño población | 200 | Balance entre diversidad y tiempo de cómputo |
| Generaciones máximas | 1000 | Suficiente para convergencia |
| Elitismo | 10 individuos | 5% de la población, preserva mejores |
| Probabilidad mutación | 20% | Balance entre exploración y explotación |
| Intentos búsqueda local | 30 | Refinamiento suficiente sin exceso |
| Tamaño torneo | 3 | Presión selectiva moderada |

---

## 7. EJEMPLO DE EJECUCIÓN

### 7.1 Datos de Entrada

**Materias**:
- Algoritmos (Prof. García, 4 horas)
- Bases de Datos (Prof. Martínez, 3 horas)
- Redes (Prof. López, 3 horas)
- Inteligencia Artificial (Prof. García, 3 horas)
- Programación Web (Prof. Martínez, 4 horas)

**Total**: 17 bloques necesarios

**Aulas**: A, B, C

**Horario**: 5 días × 6 bloques = 30 slots disponibles

### 7.2 Horario Inicial (Con Conflictos)

El horario generado aleatoriamente típicamente contiene:
- **Conflictos de aulas**: 3-5 conflictos
- **Conflictos de profesores**: 2-4 conflictos
- **Violaciones de preferencias**: 10-15 violaciones
- **Fitness inicial**: 500-800 puntos

**Ejemplo de conflictos**:
```
Lunes 8:00-9:30, Aula A:
  - Algoritmos (Prof. García)
  - Bases de Datos (Prof. Martínez)
  → CONFLICTO DE AULA

Martes 10:00-11:30:
  - Prof. García en Aula A (Algoritmos)
  - Prof. García en Aula B (IA)
  → CONFLICTO DE PROFESOR
```

### 7.3 Proceso de Optimización

```
Gen 0   | Mejor fitness: 623
Gen 50  | Mejor fitness: 145
Gen 100 | Mejor fitness: 68
Gen 150 | Mejor fitness: 23
Gen 200 | Mejor fitness: 5
Gen 215 | Mejor fitness: 0

Solución encontrada en generación 215
```

**Observaciones**:
- Rápida mejora inicial (restricciones duras)
- Refinamiento gradual (restricciones blandas)
- Convergencia a solución óptima

### 7.4 Horario Optimizado (Sin Conflictos)

**Características de la solución**:
- **Conflictos de aulas**: 0
- **Conflictos de profesores**: 0
- **Violaciones de preferencias**: 0
- **Distribución equilibrada**: ✓
- **Fitness final**: 0

**Ejemplo de horario válido**:
```
Lunes:
  8:00-9:30   | Aula A: Algoritmos (García)
  9:30-11:00  | Aula B: Bases de Datos (Martínez)
  11:00-12:30 | Aula C: Redes (López)
  
Martes:
  9:30-11:00  | Aula A: Programación Web (Martínez)
  11:00-12:30 | Aula B: IA (García)
  ...
```

---

## 8. VENTAJAS Y LIMITACIONES

### 8.1 Ventajas del Enfoque

1. **Flexibilidad**: Fácil agregar nuevas restricciones
2. **Escalabilidad**: Funciona con diferentes tamaños de problema
3. **No requiere conocimiento del dominio**: Aprende automáticamente
4. **Encuentra soluciones de calidad**: Balance entre restricciones
5. **Paralelizable**: Puede ejecutarse en múltiples cores

### 8.2 Limitaciones

1. **No garantiza óptimo global**: Heurística, no algoritmo exacto
2. **Tiempo de ejecución variable**: Depende de complejidad
3. **Requiere ajuste de parámetros**: Para mejores resultados
4. **Soluciones no determinísticas**: Diferentes corridas → diferentes resultados

---

## 9. POSIBLES EXTENSIONES

### 9.1 Lógica Difusa para Preferencias

Implementar conjuntos difusos para preferencias más flexibles:

```python
# En lugar de preferencias binarias (sí/no)
preferencias_difusas = {
    "muy_preferido": 1.0,
    "preferido": 0.7,
    "neutral": 0.5,
    "no_preferido": 0.3,
    "muy_no_preferido": 0.0
}
```

**Beneficio**: Preferencias graduales en lugar de absolutas

### 9.2 Restricciones Adicionales

- Prerequisitos entre materias
- Capacidad de aulas según matrícula
- Disponibilidad de equipos especiales
- Horarios de laboratorio
- Distancia entre aulas consecutivas

### 9.3 Optimización Multi-Objetivo

Usar NSGA-II para optimizar simultáneamente:
- Minimizar conflictos
- Maximizar satisfacción de profesores
- Minimizar ventanas libres para estudiantes
- Balancear carga de trabajo

### 9.4 Aprendizaje de Parámetros

Implementar auto-ajuste de:
- Probabilidad de mutación adaptativa
- Tamaño de población dinámica
- Presión selectiva variable

---

## 10. CONCLUSIONES

El sistema desarrollado demuestra la efectividad de los Algoritmos Genéticos para resolver problemas complejos de optimización combinatoria como la asignación de horarios académicos.

### Logros Principales

1. **Modelación correcta**: Representación adecuada del problema
2. **Restricciones implementadas**: Duras y blandas funcionando
3. **Convergencia efectiva**: Encuentra soluciones óptimas
5. **Visualización útil**: Comparación antes/después

### Coherencia del Diseño

- La representación del individuo permite operadores genéticos naturales
- La función fitness refleja correctamente el objetivo
- Los operadores mantienen factibilidad de soluciones
- El enfoque híbrido mejora convergencia
- Los parámetros están balanceados

### Aplicabilidad Práctica

Este sistema puede adaptarse fácilmente a:
- Universidades con múltiples carreras
- Colegios con diferentes niveles
- Centros de formación con cursos diversos
- Instituciones con recursos limitados

---

## 11. REFERENCIAS CONCEPTUALES

- **Algoritmos Genéticos**: Holland (1975), Goldberg (1989)
- **Optimización Combinatoria**: Problema NP-completo
- **Técnicas Híbridas**: Combinación AG + Búsqueda Local
- **Scheduling Problems**: Casos de aplicación en educación

---

## 12. INSTRUCCIONES DE USO

### Ejecutar el Sistema

```bash
python main.py
```

### Salida Esperada

1. Información del problema
2. Horario inicial con conflictos visualizado
3. Progreso del algoritmo genético
4. Horario optimizado sin conflictos
5. Análisis comparativo de mejoras

### Personalización

Para adaptar a tu institución, modificar en `main.py`:
- Lista `MATERIAS`: agregar tus materias
- Lista `PROFESORES`: agregar tus profesores con preferencias
- Lista `AULAS`: agregar tus aulas disponibles
- Constantes `DIAS` y `BLOQUES_POR_DIA` según necesidad

---

**Fecha**: Enero 2026  
**Autor**: Sistema de Optimización de Horarios  
**Versión**: 1.0
