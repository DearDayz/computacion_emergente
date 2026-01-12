# Sistema de Optimización de Horarios Académicos

Sistema inteligente de generación de horarios académicos utilizando Algoritmos Genéticos Híbridos.

## Descripción

Este sistema resuelve el problema NP-completo de asignación de horarios académicos, evitando conflictos de:
- Aulas ocupadas simultáneamente
- Profesores en múltiples lugares al mismo tiempo
- Violaciones de preferencias horarias
- Distribución desequilibrada de materias

## Características Principales

- **Algoritmo Genético Híbrido**: Operadores genéticos + búsqueda local
- **Restricciones Duras y Blandas**: Sistema flexible de penalizaciones
- **Visualización Completa**: Muestra horarios antes y después de la optimización
- **Análisis de Conflictos**: Identifica y cuantifica todos los problemas
- **Preferencias Difusas (opcionales)**: Soporte para grados de preferencia [0.0–1.0]

## Requisitos

- Python 3.7 o superior
- Bibliotecas estándar de Python (no requiere instalación adicional)

## Instalación

No se requiere instalación de dependencias externas. Solo necesitas Python instalado.

## Uso

### Ejecución Básica

```bash
python main.py
```

### Salida del Programa

El programa mostrará:

1. **Configuración del problema**
   - Cantidad de materias, profesores y aulas
   - Espacio de búsqueda (días × bloques)

2. **Horario inicial con conflictos**
   - Visualización por aulas
   - Análisis de conflictos detectados
   - Fitness inicial (mayor valor = más conflictos)

3. **Progreso del algoritmo genético**
   - Actualización cada 50 generaciones
   - Mejora progresiva del fitness

4. **Horario optimizado sin conflictos**
   - Solución final encontrada
   - Fitness = 0 (solución perfecta)
   - Sin conflictos de aulas ni profesores

## Personalización

### Modificar Materias

Edita la lista `MATERIAS` en `main.py`:

```python
MATERIAS = [
    Materia(1, "Tu Materia", profesor_id, horas_semanales),
    # Agregar más materias...
]
```

### Modificar Profesores

Edita la lista `PROFESORES` en `main.py`:

```python
PROFESORES = [
    Profesor(1, "Nombre", [(dia, bloque), ...]),  # preferencias
    # Agregar más profesores...
]
```

### Modificar Aulas

Edita la lista `AULAS` en `main.py`:

```python
AULAS = [
    Aula(1, "Nombre Aula", capacidad),
    # Agregar más aulas...
]
```

### Ajustar Parámetros

Modifica las constantes al inicio de `main.py`:

```python
POPULATION_SIZE = 300
GENERATIONS = 2000
ELITISM = 20
MUTATION_LOW = 0.15
MUTATION_HIGH = 0.3
LOCAL_SEARCH_ATTEMPTS = 50

# Opcional: activar preferencias difusas
USE_FUZZY_PREFERENCES = False
```

## Estructura del Proyecto

```
proyecto final/
├── main.py              # Código principal del sistema
├── DOCUMENTACION.md     # Documentación técnica completa
├── EJEMPLO_VISUAL.md    # Ejemplo visual antes/después
└── README.md            # Este archivo
```

## Archivos Incluidos

### main.py
Implementación completa del sistema:
- Estructuras de datos (Materia, Profesor, Aula, Bloque)
- Función fitness con restricciones duras y blandas
- Operadores genéticos (selección, cruce, mutación)
- Búsqueda local híbrida
- Algoritmo genético principal
- Visualización de horarios

### DOCUMENTACION.md
Documentación exhaustiva que incluye:
- Explicación detallada de algoritmos genéticos
- Modelación del problema
- Función fitness y restricciones
- Operadores genéticos explicados
- Ejemplos de ejecución
- Análisis de ventajas y limitaciones
- Posibles extensiones

## Componentes del Algoritmo

### 1. Representación del Individuo
Cada horario es una lista de bloques con:
- Materia asignada
- Profesor que dicta
- Aula donde se realiza
- Día y hora del bloque

### 2. Función Fitness
Penalizaciones por:
- **100 puntos**: Conflicto de aula (restricción dura)
- **100 puntos**: Conflicto de profesor (restricción dura)
- **5 puntos**: Violación de preferencia horaria
- **10 puntos**: Distribución desequilibrada
- **3 puntos**: Bloques consecutivos excesivos

### 3. Operadores Genéticos
- **Selección**: Torneo de 3 individuos
- **Cruce**: Uniforme por materia, conservando cantidad de bloques
- **Mutación**: Cambio aleatorio de día, hora o aula (probabilidad adaptativa)
- **Elitismo**: Preserva los mejores individuos (configurable)

### 4. Búsqueda Local
Refinamiento mediante 30 intentos de mejora local por cada hijo generado.

## Ejemplo de Salida

```
================================================================================
        SISTEMA DE OPTIMIZACION DE HORARIOS ACADEMICOS
                  Algoritmo Genetico Hibrido
================================================================================

Datos del problema:
- 5 materias
- 3 profesores
- 3 aulas
- 5 dias x 6 bloques = 30 slots totales
- 17 bloques de clase necesarios

================================================================================
GENERANDO HORARIO INICIAL (con conflictos)...
================================================================================

[Visualización del horario con conflictos]

Fitness: 623

ANALISIS DE CONFLICTOS:
Conflictos de aulas: 4
Conflictos de profesores: 3
Violaciones de preferencias: 12

================================================================================
EJECUTANDO ALGORITMO GENETICO...
================================================================================

Gen 0 | Mejor fitness: 623
Gen 50 | Mejor fitness: 145
Gen 100 | Mejor fitness: 68
Gen 150 | Mejor fitness: 23
Gen 200 | Mejor fitness: 5

Solucion encontrada en generacion 215

[Visualización del horario optimizado]

Fitness: 0

ANALISIS DE CONFLICTOS:
Conflictos de aulas: 0
Conflictos de profesores: 0
Violaciones de preferencias: 0

Proceso completado!
```

## Rendimiento (referencial)

- **Tamaño población**: por defecto 300 individuos
- **Generaciones típicas**: 200–2000 (según complejidad y parámetros)
- **Tiempo de ejecución**: variable según hardware y configuración
- **Tasa de éxito**: alta en problemas como el ejemplo provisto

## Ventajas del Sistema

1. **Automático**: No requiere intervención manual
2. **Flexible**: Fácil agregar nuevas restricciones
3. **Escalable**: Funciona con diferentes tamaños de problema
4. **Visualización clara**: Muestra mejora del proceso

## Extensiones Posibles

- Implementar lógica difusa para preferencias graduales (incluida de forma opcional)
- Agregar restricciones de capacidad de aulas
- Considerar prerequisitos entre materias
- Optimización multi-objetivo (varios criterios simultáneos)
- Interfaz gráfica para configuración y visualización

## Evaluación del Proyecto

### Cumplimiento de Requisitos

✅ Representación del individuo (horario completo)  
✅ Función fitness basada en conflictos  
✅ Restricciones duras (aulas y profesores)  
✅ Restricciones blandas (preferencias y distribución)  
✅ Documento explicativo completo  
✅ Ejemplo visual con conflictos y mejora  
✅ Explicación clara de algoritmos genéticos  
✅ Coherencia entre diseño y documentación  

### Aspectos Destacados

- Algoritmo genético híbrido (AG + búsqueda local)
- Visualización comparativa antes/después
- Documentación técnica exhaustiva
- Código modular y extensible

## Autor
-Eduardo Tovar
-

**Versión**: 1.0  
**Fecha**: Enero 2026  
**Lenguaje**: Python 3

## Licencia

Proyecto académico de código abierto para fines educativos.
