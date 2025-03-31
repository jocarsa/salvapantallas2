package proyectojava21;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import javax.imageio.ImageIO;


public class ProyectoJava21 {

    
    public static void main(String[] args) {
        // TODO code application logic here
        try{
            Class.forName("com.mysql.jdbc.Driver");
            // Ahora establezco la conexión
            Connection conexion = DriverManager.getConnection("jdbc:mysql://localhost:8889/cursojava", "cursojava", "cursojava");
            // Preparo una petición a la base de datos
            Statement peticion = conexion.createStatement();
            // A continuación le pedimos algo a la base de datos y lo guardamos dentro de un objeto (como si fuera una variable)
            ResultSet resultado = peticion.executeQuery("SELECT * FROM cursos");
            int numero = 1;
            // Mientras que el resultado tenga lineas
            while(resultado.next()){
                // Imprimeme en pantalla el resultado
                System.out.println(resultado.getString(3));
                //////////////////////////////////////////////////////////////////////////////
                
                int anchura = 800;                                                      // Anchura que tendrá la imagen
                int altura = 400;                                                       // Altura que tendrá la imagen

                                                         // Recurso vacío por si más adelante nos interesa             

                BufferedImage imagencacheada = new BufferedImage(anchura,altura,BufferedImage.TYPE_INT_RGB);    // Creo una imagen con su altura, su anchura, y el tipo de color

                Graphics2D graficos = imagencacheada.createGraphics();                  // Digo que dentro de esa imagen voy a pintar cosas
                // /////////////////EN ESTE TROZO PUEDES PINTAR////////////////////////////
                graficos.setColor(Color.white);
                graficos.fillRect(0,0,anchura,altura);

                

                

                BufferedImage imagen = null;   
                imagen = ImageIO.read(new File("logos/"+resultado.getString(7)));
                graficos.drawImage(imagen, 0, 0, 400,400, null);
                
                BufferedImage imagen2 = null;   
                imagen2 = ImageIO.read(new File("fotos/Fotos Jose Vicente Carratala "+String.format("%05d",numero)+".jpg"));
                
                graficos.drawImage(imagen2, 400, 0, 400,400, null);
                
                graficos.setColor(Color.white);                                           // Digo que lo que voy a pintar a continuación es con color rojo
                graficos.fillRect(390, 0, 20, 400);                                    // Pinto un rectangulo
                Color negrotransparente = new Color(0, 0, 0, 127);
                graficos.setColor(negrotransparente);                                           // Digo que lo que voy a pintar a continuación es con color rojo
                graficos.fillRect(0, 370, anchura, 400);
                
                graficos.setColor(Color.white);
                graficos.setFont(new Font("Arial", Font.PLAIN, 28));
                graficos.drawString(resultado.getString(3), 10, 395);
                


                // /////////////////EN ESTE TROZO PUEDES PINTAR////////////////////////////
                graficos.dispose();                                                     // Libero el recurso

                File archivo = new File("guardado/"+String.format("%05d",numero)+""+resultado.getString(2)+".png");                           // Apunto a un nuevo archivo

                ImageIO.write(imagencacheada,"png",archivo);                            // Con la libreria correspondiente, guardo el png en ese archivo
                numero++;
                
                //////////////////////////////////////////////////////////////////////////////
            }
        }catch(Exception e){
            e.printStackTrace();
        }
    }
    
}
