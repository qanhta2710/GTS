#include <stdio.h>
#include <math.h>

double func(double x) {
    return pow(x, 5) - 3 * pow(x, 3) + 2 * pow(x, 2) - x + 5; // Nhap phuong trinh f(x)
}

// CAM DONG VAO DAY
void printSolution(double (*func)(double), double a, double b, double relativeError) {
    int i = 0;
    printf("%3s %15s %15s %15s %15s %15s\n", "i", "a", "b", "c", "f(c)", "Error");
    double c = (a + b) / 2; // initial value
    double z = func(c);
    double currentError = fabs((b - a)) / fabs(c);
    printf("%3d %15.10lf %15.10lf %15.10lf %15.10e %c\n", 0, a, b, c, z, '\0');
    while (currentError > relativeError) {
        if (func(c) == 0) {
            printf("Solution: %lf\n", c);
        } else {
            if (func(a) * z > 0) {
                a = c;
            } else {
                b = c;
            }
        }
        double tmp = c;
        c = (a + b) / 2;
        currentError = fabs(c - tmp) / fabs(c);
        i++;
        printf("%3d %15.10lf %15.10lf %15.10lf %15.10lf %15.10e\n", i, a, b, c, func(c), currentError);
    }
}
int main() {
    double a = -2.5; // Nhap khoang cach li a
    double b = -2; // Nhap khoang cach li b
    double relativeError = 0.05 / 100; // Nhap sai so tuong doi
    printSolution(func, a, b, relativeError);
    return 0;
}
