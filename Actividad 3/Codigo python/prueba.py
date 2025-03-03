# Criterio de parada: Si el cambio entre iteraciones es menor que la tolerancia, detener
        if x_anterior is not None:
            cambio = max(abs(x_nuevo[j] - x_anterior[j]) for j in range(len(x_nuevo)))
            if cambio < tolerancia:
                print(f"Convergencia alcanzada en iteración {i + 1} con cambio {cambio:.2e}")
                break