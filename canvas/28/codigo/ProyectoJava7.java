
package proyectojava7;


public class ProyectoJava7 {

    
    public static void main(String[] args) {
        int edad = 2;
        
        if(edad < 20){
            if(edad < 10){
                 System.out.println("Eres un niño");
            }else{
                System.out.println("Eres un adolescente");
            }
        }else{
            if(edad < 30){
                 System.out.println("Eres un joven");
            }else{
                System.out.println("Ya no eres un joven");
            }
           
        }
    }
    
}
