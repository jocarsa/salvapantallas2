package proyectojava23;


public class Persona {
// Voy a declarar propiedades de la clase
    double minx = 0;                                                      // Establezco un minimo
    double maxx = 1920;                                                       // Establezco un maximo
    double randomx = minx + Math.random() * (maxx - minx);                      // Creo un numero aleatorio entre el minimo y el maximo
    public double x = randomx;
    double miny = 0;                                                      // Establezco un minimo
    double maxy = 1080;                                                       // Establezco un maximo
    double randomy = miny + Math.random() * (maxy - miny);                      // Creo un numero aleatorio entre el minimo y el maximo
    public double y = randomy;
    public float direccion = 0;
    
    public void mueveBola(){                                                    // Esta funcion mueve la bola
        double min = -0.5;                                                      // Establezco un minimo
        double max = 0.5;                                                       // Establezco un maximo
        double random = min + Math.random() * (max - min);                      // Creo un numero aleatorio entre el minimo y el maximo
        direccion += random;                                                    // Vario la direccion de forma aleatoria
        x += Math.cos(direccion);                                               // Aumento la x en base a la direccion y su coseno
        y += Math.sin(direccion);                                               // Aumento la y en base a la direccion y su seno
        if(x > 1920){direccion += Math.PI;}                                      // En el caso de que la x sea menor que 400, pega la vuelta
        if(x < 0){direccion += Math.PI;}                                        // Pega la vuelta al colisionar
        if(y > 1080){direccion += Math.PI;}                                      // Pega la vuelta al colisionar
        if(y < 0){direccion += Math.PI;}                                        // Pega la vuelta al colisionar
    }
    
}
