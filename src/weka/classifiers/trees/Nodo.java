package weka.classifiers.trees;
/**
 * Un nodo para un árbol de desición
 */

import java.util.ArrayList;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Attribute;


public class Nodo {
    Attribute atributo;
    String rama;
    String valor;
    ArrayList<Nodo> hijos;

    Nodo(Attribute at) { // arbol
        atributo = at;
        valor = "";
        hijos = new ArrayList<Nodo>();
    }

    Nodo(String val) { // Hoja
        atributo = null;
        valor = val;
        hijos = null;
    }
 
    void AgregaHijo(Nodo hijo, String val) {
        // Para llegar a hijo se necesita el valor rama
        Nodo nuevo = hijo.clone();
        nuevo.rama = val;
        hijos.add(nuevo);

    }

    Nodo eligeHijo(String val) {
        for (Nodo h : hijos) {
            // Elige el hijo cuya rama sea <val>
            if (h.rama.equals(val))
            return h;
        }
        // No debe llegar aca
        return null;
        
    }

    String Evalua(Instance d) {
        Nodo n = this;
        while (!n.esHoja()) {
            n = n.eligeHijo(d.stringValue(n.atributo));
        }
        String val = n.valor;

        return val;
    }

    Boolean esHoja() {
        return (hijos == null);
    }

    @Override
    public Nodo clone() {
        // Arbol o hoja?
        Nodo n;
        if (esHoja()) {
            n = new Nodo(valor);
            n.rama = rama;
        }
        else {
            n = new Nodo(atributo);
            n.rama = rama;
            n.hijos = hijos;
        }

        return n;

    }

    @Override
    public String toString() {
        String res = cadena(1);
        return res;
    }
    
    public String cadena(int p) {
        /* La tipica
            Nodo
              hijo1
              hijo2
              ...
        */ 
        String san = "";
        String sanp = ""; // Sangria previa
        for (int i = 0; i < p*2; i++) {
            san+=" ";
        }
        for (int i = 0; i < (p-1)*2; i++) {
            sanp+=" ";
        }
        String res = sanp;
        if (hijos != null) {
            res += atributo.name();
            for (Nodo h : hijos) {
                res += "\n" + san + h.rama + "\n" + sanp + " " + h.cadena(p+1);
            }
        }
        else {
            res += valor;
        }
        return res;
    }

    public static void main(String[] args) throws Exception {
        // Lee datos de tenis
		DataSource source = new DataSource("weather.nominal.arff");
		Instances datos = source.getDataSet();
        // Crea arbol Tenis
        Nodo si = new Nodo("Si");
        Nodo no = new Nodo("No");
        Nodo hum = new Nodo(datos.get(0).attribute(2)); // El 2 es humedad
        hum.AgregaHijo(no, "high");
        hum.AgregaHijo(si, "normal");
        Nodo wind = new Nodo(datos.get(0).attribute(3)); // El 3 es wind
        wind.AgregaHijo(no, "TRUE");
        wind.AgregaHijo(si, "FALSE");
        // Nodo raiz
        Nodo raiz = new Nodo(datos.get(0).attribute(0)); // 0 es outlook
        raiz.AgregaHijo(hum, "sunny");
        raiz.AgregaHijo(si, "overcast");
        raiz.AgregaHijo(wind, "rainy");

        System.out.println(raiz);

        for (Instance ins : datos) {
            String c = raiz.Evalua(ins);
            System.out.println(ins.toString() + "->" + c );
        }
        
    }
}

