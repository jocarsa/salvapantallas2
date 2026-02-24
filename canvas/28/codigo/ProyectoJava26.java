
package proyectojava26;

import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class ProyectoJava26 {

    
    public static void main(String[] args) {
       // TODO code application logic here
       String correo = "^[A-Z0-9._+-]+@[A-Z0-9.-]+\\.[A-Z]{2,6}$";
        Pattern patron = Pattern.compile(correo,Pattern.CASE_INSENSITIVE);
        Matcher frase = patron.matcher("jocarsa%2@gmail.com");
        
        boolean encontrado = frase.find();
        
        if(encontrado){
            System.out.println("Esto si que es un email");
        }else{
            System.out.println("Esto no es un email");
        }
    }
    
}
