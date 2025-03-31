package proyectojava20;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class ProyectoJava20 {

    
    public static void main(String[] args) throws IOException {
       
        
        int anchura = 800;                                                      // Anchura que tendrá la imagen
        int altura = 400;                                                       // Altura que tendrá la imagen
        
                                                 // Recurso vacío por si más adelante nos interesa             
        
        BufferedImage imagencacheada = new BufferedImage(anchura,altura,BufferedImage.TYPE_INT_RGB);    // Creo una imagen con su altura, su anchura, y el tipo de color
        
        Graphics2D graficos = imagencacheada.createGraphics();                  // Digo que dentro de esa imagen voy a pintar cosas
        // /////////////////EN ESTE TROZO PUEDES PINTAR////////////////////////////
        graficos.setColor(Color.white);
        graficos.fillRect(0,0,anchura,altura);
        
        graficos.setColor(Color.RED);                                           // Digo que lo que voy a pintar a continuación es con color rojo
        graficos.fillRect(20, 20, 300, 300);                                    // Pinto un rectangulo
        
        graficos.setColor(Color.green);
        graficos.drawString("Programa de Jose Vicente", 300, 200);
        
        BufferedImage imagen = null;   
        imagen = ImageIO.read(new File("logos/logo_java.png"));
        graficos.drawImage(imagen, 0, 0, 400,400, null);
        
       
        
        
        // /////////////////EN ESTE TROZO PUEDES PINTAR////////////////////////////
        graficos.dispose();                                                     // Libero el recurso
        for(int i = 0;i<10;i++){
            File archivo = new File("guardado/primeraprueba"+i+".png");                           // Apunto a un nuevo archivo
        
            ImageIO.write(imagencacheada,"png",archivo);                            // Con la libreria correspondiente, guardo el png en ese archivo
        }
        
        
    }
    
}
