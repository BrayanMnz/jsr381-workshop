package brayanmnz.jsr381.workshop.examples;

import javax.visrec.spi.ServiceProvider;
import javax.visrec.spi.ServiceRegistry;

public class ImplementationExample {


    public static void main(String[] args){
    
        String IMPLEMENTATION_TEXT = "VisRec API (JSR 381) currently being used is " + ServiceProvider.current().getImplementationService().toString();
        System.out.println(IMPLEMENTATION_TEXT);

    }


}
