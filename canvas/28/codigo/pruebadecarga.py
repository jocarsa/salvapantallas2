import time

tiempoinicio = int(round(time.time()*1000))
contador = 0;
numero = 0.0000004234
while contador < 1000000000:
    numero *= 1.000001
    contador+=1
tiempofinal = int(round(time.time()*1000))
print(tiempofinal-tiempoinicio)
