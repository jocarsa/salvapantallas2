
package proyectojava15;


public class ProyectoJava15 {

    
    public static void main(String[] args) {
        String agenda[][] = {{"Juan","Jose","Jorge","Julia"},{"1234","3456","5678","7890"}};
        for(int i = 0;i<agenda[0].length;i++){
            System.out.println("Que sepas que el nombre de este contacto es: "+agenda[0][i]);
            System.out.println("Que sepas que el telefono de este contacto es: "+agenda[1][i]);
            System.out.println("//////////////////////////");
        }
    }
    
}
