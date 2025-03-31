
package proyectojava13;


public class ProyectoJava13 {

    
    public static void main(String[] args) {
        // TODO code application logic here
        Persona juan = new Persona();
        Persona jorge = new Persona();
        System.out.println("El nombre ahora mismo de Juan es: "+juan.nombre);
        juan.nombre = "Juan";
        juan.edad = 35;
        jorge.nombre = "Jorge";
        jorge.edad = 22;
        /*
        System.out.println("El nombre ahora mismo de Juan es: "+juan.nombre);
        System.out.println("La edad ahora mismo de Juan es: "+juan.edad);
        System.out.println("El nombre ahora mismo de Jorge es: "+jorge.nombre);
        System.out.println("La edad ahora mismo de Jorge es: "+jorge.edad);
        */
        juan.saluda();
        jorge.saluda();
        
    }
    
}
