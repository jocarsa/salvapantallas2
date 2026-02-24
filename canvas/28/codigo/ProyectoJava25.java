
package proyectojava25;

import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class ProyectoJava25 {

    
    public static void main(String[] args) {
        // TODO code application logic here
        Pattern patron = Pattern.compile("coco",Pattern.CASE_INSENSITIVE);
        Matcher frase = patron.matcher("El zumo que me he puesto tiene Coco y piña");
        
        boolean encontrado = frase.find();
        
        if(encontrado){
            System.out.println("Si que se ha encontrado");
        }else{
            System.out.println("No se ha encontrado");
        }
    }
    
}
