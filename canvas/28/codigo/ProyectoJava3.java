package proyectojava3;

public class ProyectoJava3 {

    
    public static void main(String[] args) {
        
        float operador1 = 4;
        float operador2 = 3;
        
        // Suma
        float suma = operador1 + operador2;
        System.out.println("La suma es: "+suma);
        // Resta
        float resta = operador1 - operador2;
        System.out.println("La resta es: "+resta);
        // Multiplicacion
        float multiplicacion = operador1 * operador2;
        System.out.println("La multiplicación es: "+multiplicacion);
        // División
        double division = operador1 / operador2;
        System.out.println("La división es: "+division);
        // Operador comparacion de igualdad
        boolean igualdad = operador1 == operador2;
        System.out.println("La comparación es: "+igualdad);
        // Operador comparacion de no igualdad
        boolean noigualdad = operador1 != operador2;
        System.out.println("La comparación es: "+noigualdad);
        // Operador comparacion de menor que
        boolean menorque = operador1 < operador2;
        System.out.println("La comparación es: "+menorque);
        // Operador comparacion de menor que
        boolean mayorque = operador1 > operador2;
        System.out.println("La comparación es: "+mayorque);
        
    }
    
}
