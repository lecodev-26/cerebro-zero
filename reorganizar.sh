#!/bin/bash

# Crear carpetas
mkdir -p cerebros entrenadores datos modelos_guardados utils scripts pruebas

# Mover archivos
mv red_neuronal.py cerebros/ 2>/dev/null
mv red_neuronal_tanh.py cerebros/ 2>/dev/null
mv red_neuronal_relu.py cerebros/ 2>/dev/null
mv red_neuronal_premium.py cerebros/ 2>/dev/null
mv red_profunda.py cerebros/ 2>/dev/null
mv memoria.py cerebros/ 2>/dev/null

mv entrenar.py entrenadores/ 2>/dev/null
mv entrenar_xor.py entrenadores/ 2>/dev/null
mv entrenar_profundo.py entrenadores/ 2>/dev/null
mv entrenar_memoria.py entrenadores/ 2>/dev/null
mv entrenar_premium.py entrenadores/ 2>/dev/null

mv datos.py datos/ 2>/dev/null
mv datos_reales.py datos/ 2>/dev/null
mv datos_reales_tanh.py datos/ 2>/dev/null
mv datos_reales_relu.py datos/ 2>/dev/null
mv seno_polinomico.py datos/ 2>/dev/null

mv visualizar.py utils/ 2>/dev/null
mv autoencoder.py utils/ 2>/dev/null
mv convolutional.py utils/ 2>/dev/null
mv reforzamiento.py utils/ 2>/dev/null

mv todo_en_uno.py scripts/ 2>/dev/null
mv cargar_cerebro.py scripts/ 2>/dev/null

mv pruebas.py pruebas/ 2>/dev/null
mv *.pkl modelos_guardados/ 2>/dev/null

echo "✅ Archivos reorganizados!"
