package proyectojava12;


public class ProyectoJava12 {

    
    public static void main(String[] args) {
        // TODO code application logic here
        saluda("Jose Vicente");
        saluda("Juan");
        saluda("Jorge");
        saluda("Jaime");
        saluda("Parvin");
        saluda();
        saluda("Julia","lunes");
    }
    
    public static void saluda(String nombre){
        System.out.println("Hola, "+nombre+", como estas?");
    }
    
    public static void saluda(){
        System.out.println("Hola, como estas?");
    }
    public static void saluda(String nombre,String dia){
        System.out.println("Hola, "+nombre+", como estas? Sabes que hoy es "+dia+"?");
    }
    
    
}
