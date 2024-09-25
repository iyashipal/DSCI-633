import java.util.Scanner;
public class WeightConverter {
    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
        float lbs=sc.nextFloat();
        //1kg=2.2pounds
        float kgs=(float)(lbs/2.2);
        System.out.printf("kgs=%.2f\n",kgs);

    }

}
