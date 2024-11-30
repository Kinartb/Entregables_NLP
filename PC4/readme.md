# PC4
ARTURO HINOSTROZA OLIVERA
## _Ajustar finamente un LLM con PPO vs DPO vs ORPO utilizando el paquete PEFT_

El ajuste fino de modelos de lenguaje a gran escala (LLMs) es una tarea esencial para adaptar modelos preentrenados a aplicaciones específicas. En este proyecto, se propone comparar tres técnicas de optimización por políticas: Proximal Policy Optimization (PPO), Deterministic Policy Optimization (DPO) y Optimistic Regularized Policy Optimization (ORPO), utilizando el paquete PEFT (Parameter-Efficient Fine-Tuning). El objetivo es evaluar cuál de estos métodos ofrece el mejor rendimiento en términos de eficiencia y precisión, especialmente en entornos con recursos computacionales limitados.

## Objetivos
Implementar  los métodos de optimización: 

- Policy Optimization (PPO)
- Deterministic Policy Optimization (DPO)
- Optimistic Regularized Policy Optimization (ORPO)

Utilizando PEFT para optimizar el ajuste fino.

Métricas a evaluar:

- Evaluar cada modelo ajustado en tareas específicas
- Precisión
- Tiempo de entrenamiento 
- Eficiencia de recursos.

## Recursos

Estamos usando los siguientes recursos:

- Transformers: Carga modelos preentrenados como GPT-2.
- GPT-2: es un modelo de transformadores entrenado previamente en un corpus muy grande de datos en inglés de manera autosupervisada.
- wikitext - Dataset a trabajar que se divide en 
-- wikitext_test
-- wikitext_train
-- wikitext_validation
- PEFT  - Parameter-Efficient Fine-Tuning
- LORA - LowRank Adaptation


## Instalación

Instala las siguientes dependencias para ser utilizadas.

```sh
pip install transformers datasets peft
pip install transformers torch accelerate
```
    
## Desarrollo

Se tiene el código en el siguiente enlace:

## Elección del dataset:

El dataset WikiText fue seleccionado para este proyecto debido a sus características específicas que lo hacen adecuado para entrenar y ajustar modelos de lenguaje como GPT-2 utilizando técnicas como PEFT. WikiText está compuesto por textos extraídos de artículos de Wikipedia, lo que asegura una gramática con pocas irregularidades. Esto permite que los modelos de lenguaje se centren en aprender patrones lingüísticos y estructuras complejas sin distraerse con ruido o inconsistencias típicas de otros datasets.

El dataset está diseñado específicamente para tareas de modelado de lenguaje. Esto incluye generación de texto, predicción de palabras y comprensión de relaciones contextuales, lo que lo hace ideal para evaluar el rendimiento de los modelos ajustados.

## Aspectos teóricos:

### Problemática:

El ajuste fino sigue siendo esencial para mejorar el rendimiento de LLM en tareas y conjuntos de datos de usuarios no vistos. Con el tamaño del modelo en aumento (por ejemplo, 1.500 millones en GPT-2 a 175.000 millones en GPT-3), el paradigma estándar de ajuste fino completo requiere miles de trabajos de GPU en paralelo, lo que es altamente ineficiente e insostenible. 

### Fine-Tuning:
Se ha desarrollado un tipo de algoritmo, denominado ajuste fino con parámetros eficientes (PEFT), que tiene como objetivo ajustar parámetros mínimos para lograr un mejor rendimiento en comparación con el ajuste completo en tareas posteriores. El ajuste fino con eficiencia de parámetros nos permite reutilizar modelos entrenados previamente y, al mismo tiempo, minimizar el consumo de recursos y de recursos. En resumen, el ajuste fino con eficiencia de parámetros es útil por al menos cinco razones:

- Costos computacionales reducidos (requiere menos GPU y menos tiempo de GPU);
- Tiempos de entrenamiento más rápidos (finaliza el entrenamiento más rápido);
- Requerimientos de hardware más bajos (funciona con GPU más pequeñas y menos memoria);
- Mejor rendimiento de modelado (reduce el sobreajuste);
- Menos almacenamiento (la mayoría de los pesos se pueden compartir entre diferentes tareas).   

### Tipos de Fine-Tuning:
Se diferencian entre sí en términos de los diferentes módulos o parámetros ajustables adicionales.
- **PEFT aditiva**: Que modifica la arquitectura del modelo inyectando nuevos módulos entrenables o parámetros.
- **PEFT selectiva** : Que hace que un subconjunto de parámetros sea entrenable durante el ajuste fino.
- **PEFT reparametrizada**: Que construye una reparametrización (de baja dimensionalidad) de los parámetros del modelo original para el entrenamiento y luego la transforma de manera equivalente para la inferencia.
- **PEFT híbrida**: Que combina las ventajas de diferentes métodos PEFT para construir un modelo PEFT unificado. 

### PEFT reparametrizada:

La reparametrización consiste en transformar de manera equivalente la arquitectura de un modelo de una a otra mediante la transformación de sus parámetros. En el contexto de PEFT, esto suele significar construir una parametrización de bajo rango para lograr el objetivo de eficiencia de parámetros durante el entrenamiento. Para la inferencia, el modelo se puede convertir a su parametrización de peso original, lo que garantiza una velocidad de inferencia sin cambios.

### LORA 

LORA está diseñado para ajustar modelos a gran escala de manera eficiente al enfocarse en un pequeño subconjunto de los pesos del modelo que tienen el impacto más significativo en la tarea en cuestión. Esto contrasta con el ajuste fino tradicional, donde se pueden actualizar muchos más pesos. LORA logra esto al:

Rastrear los cambios en los pesos en lugar de actualizarlos directamente.
Descomponer matrices grandes de cambios de peso en matrices más pequeñas que contienen los "parámetros entrenables".

Este enfoque ofrece varias ventajas:

-	Reducción significativa de los parámetros entrenables, lo que permite un ajuste fino más rápido y eficiente.
-	Conservación de los pesos originales entrenados previamente, lo que permite múltiples modelos livianos para diferentes tareas.
-	Compatibilidad con otros métodos eficientes en cuanto a parámetros, lo que permite una mayor optimización.
-	Rendimiento comparable a los modelos totalmente ajustados en muchos casos.
-	No hay latencia de inferencia adicional, ya que los pesos del adaptador se pueden fusionar con el modelo base.
-	LoRA reduce y acelera el ajuste fino mediante la descomposición de matrices


### Arquitectura LORA
LoRA introduce dos matrices de peso entrenables, $$W_{\text{up}} \in \mathbb{R}^{d \times r}$$ y $$W_{\text{down}} \in \mathbb{R}^{r \times k}$$, donde el rango $$r$$ satisface $$r < \min(d, k)$$, operando en paralelo a $$W_0$$. Dejemos que $$h_{\text{in}}$$ represente la entrada.

En condiciones normales, la salida a través de $$W_0$$ es:

$$h_{\text{out}} = W_0 h_{\text{in}}$$

En cambio, LoRA modifica esta salida introduciendo una implementación adicional $$\Delta W$$ que encapsula el conocimiento específico de la tarea:

$$h_{\text{out}} = W_0 h_{\text{in}} + \frac{\alpha}{r} \Delta W h_{\text{in}} = W_0 h_{\text{in}} + \frac{\alpha}{r} W_{\text{up}} W_{\text{down}} h_{\text{in}}$$

Donde:
- $$\alpha$$ denota un factor de escala.

 Inicialización:
1. $$W_{\text{down}}$$ se inicializa utilizando una distribución gaussiana aleatoria.
2. $$W_{\text{up}}$$ se inicializa en cero, garantizando que $$\Delta W$$ inicialmente tenga un valor de cero.

LoRA es sencillo de implementar y ha sido evaluado en modelos con hasta **175 mil millones de parámetros**. Una vez completado el ajuste fino, los pesos adaptativos de LoRA se integran perfectamente con los pesos preentrenados del modelo base, manteniendo la eficiencia del modelo sin añadir carga adicional durante la inferencia.   
   



