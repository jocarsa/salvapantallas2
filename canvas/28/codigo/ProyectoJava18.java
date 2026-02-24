package proyectojava18;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;


public class ProyectoJava18 {

    
    public static void main(String[] args) {
        // TODO code application logic here
        try{
            Class.forName("com.mysql.jdbc.Driver");
            // Ahora establezco la conexión
            Connection conexion = DriverManager.getConnection("jdbc:mysql://localhost:8889/cursojava", "cursojava", "cursojava");
            // Preparo una petición a la base de datos
            Statement peticion = conexion.createStatement();
            // A continuación le pedimos algo a la base de datos y lo guardamos dentro de un objeto (como si fuera una variable)
            ResultSet resultado = peticion.executeQuery("SELECT * FROM agenda");
            // Mientras que el resultado tenga lineas
            while(resultado.next()){
                // Imprimeme en pantalla el resultado
                System.out.println(resultado.getString(1)+"-"+resultado.getString(2)+"-"+resultado.getString(3));
            }
        }catch(Exception e){
            e.printStackTrace();
        }
    }
    
}
