package proyectojava6;

public class ProyectoJava6 {
    
    public static void main(String[] args) {
        int edad1 = 42;
        int edad2 = 25;
        int maximo = Math.max(edad1,edad2);
        System.out.println("El maximo es: "+maximo);
        double numero = 45.2;
        double redondeo = Math.ceil(numero);
        System.out.println("El redondeo es: "+redondeo);
        double angulo = Math.PI;
        double seno = Math.sin(angulo);
        double coseno = Math.cos(angulo);
        System.out.println("El coseno es: "+coseno);
    }
    
}
