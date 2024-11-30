from fastapi import FastAPI
import avx_dot_product
import numpy as np
import time
from pydantic import BaseModel
from typing import List
import json
import random

def listaAleatorios(n: int):
      lista = [0]  * n
      for i in range(n):
          lista[i] = random.randint(0, 10000)* 0.0001
      return lista

app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]

@app.post("/avxdotproduct")
async def calculo():
    start = time.time()

    # Crear dos grandes vectores
    vec1 = np.random.rand(1000000).astype(np.float32)
    vec2 = np.random.rand(1000000).astype(np.float32)
    
    #vec1: VectorF 
    #vec2: VectorF

    #vec1 = listaAleatorios(1000000)
    #vec2 = listaAleatorios(1000000)

    # Calcular el producto punto usando el módulo AVX
    result = avx_dot_product.dot_product(vec1, vec2)
    #print("Resultado del producto punto:", result)

    # Comparar con NumPy
    expected = np.dot(vec1, vec2)
    #print("Resultado esperado (NumPy):", expected)

    
    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "vec1": vec1,
        "vec2": vec2,
        "Resultado del producto punto": result,
        "Resultado esperado (NumPy)": expected
    }
    jj = json.dumps(str(j1))

    return jj