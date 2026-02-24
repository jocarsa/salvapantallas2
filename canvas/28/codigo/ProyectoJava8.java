package proyectojava8;

public class ProyectoJava8 {

    
    public static void main(String[] args) {
        String diadelasemana = "longaniza";
        
        switch(diadelasemana){
            case "lunes":
                System.out.println("Hoy es el peor dia de la semana");break;
            case "martes":
                System.out.println("Hoy es el segundo peor dia de la semana");break;
            case "miercoles":
                System.out.println("Parece que ya estamos a mitad");break;
            case "jueves":
                System.out.println("Ya casi es viernes");break;
            case "viernes":
                System.out.println("Hoy es el mejor dia de la semana");break;
            case "sabado":
                System.out.println("Hoy es el super mejor dia de la semana");break;
            case "domingo":
                System.out.println("Parece mentira que mañana ya es lunes de nuevo");break;
            default:
                System.out.println("Yo no se lo que has escrito, pero eso no es un día");break;
        }
    }
    
}
