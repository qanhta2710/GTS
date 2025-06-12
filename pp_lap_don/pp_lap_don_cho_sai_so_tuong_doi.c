#include <stdio.h>
#include <math.h>
#define e 2.718281828459045

// Can bac ba dung ham cbrt(), can bac hai dung ham sqrt()

double phi(double x) {
    return log(10*x - 7); // Nhap ham phi(x)
}

void printSolution(double (*phi)(double), double xk, double epsilon) {
    double tmp = xk;
    int k = 0;
    printf("%3s %15s %15s\n", "k", "xk", "Relative Error");
    // Kiểm tra sai số tương đối ban đầu
    if (fabs(xk) > 1e-10 && fabs(phi(xk) - xk) / fabs(xk) <= epsilon) {
        printf("%3d %15.9lf %15.9e\n", k, xk, 0.0);
        return;
    }
    do {
        tmp = xk;
        xk = phi(xk);
        k++;
        // Tính sai số tương đối, kiểm tra xk != 0
        double relative_error = (fabs(xk) > 1e-10) ? fabs(xk - tmp) / fabs(xk) : 0.0;
        printf("%3d %15.9lf %15.9e\n", k, xk, relative_error);
    } while (fabs(xk) > 1e-10 && fabs(xk - tmp) / fabs(xk) > epsilon);
}

int main() {
    double x0 = 3; // Nhap diem khoi tao x0
    // neu phi(x) > 0 thi x0 = a hoac x0 = b
    // neu phi(x) < 0 thi x0 = alpha sao cho a < alpha < (a+b)/2 hoac x0 = beta sao cho (a+b)/2 < beta < b  
    double epsilon = 0.5 * pow(10, -7); // Nhap sai so

    printSolution(phi, x0, epsilon);
    return 0;
}