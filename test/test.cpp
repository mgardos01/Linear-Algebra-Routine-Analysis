#include <vector>
#include <cstdlib>
#include <cublas_v2.h>
#include <assert.h>
#include <fstream>
#include <string>
#include <algorithm>

#include "../src/dasum.h"
#include "../src/dnrm2.h"
#include "../src/dgemm.h"
#include "../src/daxpy.h"
#include "../src/dcopy.h"

#define EPSILON 1E-4

const std::string sep = "::~~~~~~~~~~~~~~~~~~~~~~~~~~";

double random_double() { 
    return 100 * ( (double)std::rand() / (double)RAND_MAX ); 
}

bool double_is_equal(double a, double b) {
    return ((a - b) < EPSILON) && ((b - a) < EPSILON);
}

void dasum_test() {
    std::cout << "dasum_tests" << sep << std::endl;

    std::ofstream oFile("dasum.dat");

    oFile << "n" << "\t" 
          << "my_dasum_time" << "\t"
          << "cublas_dasum_time" << "\n";

    float my_dasum_time, cublas_dasum_time, time_instance;

    for (unsigned int n = 2; n <= pow(2, 20); n = n * 2 + 1) {
        my_dasum_time = cublas_dasum_time = 0.0f;

        std::vector<double> input(n, 1.0f);
        std::vector<double> output(1, -1.0f);

        for (int i = 0; i < 10; i++) {
            time_instance = dasum(input.data(), output.data(), n);
            if (i > 0) my_dasum_time += time_instance;
            if (output.at(0) == n) {
                std::cout << "myDasum :: " << n << "-" << i << " \u2192 \u2713\n";  // good~!
            } else {
                std::cerr << "myDasum :: " << n << "-" << i << " \u2192 \u2717\n"; // bad~!
            }
        }
        my_dasum_time /= 9;

        for (int i = 0; i < 10; i++) {
            time_instance = cublas_dasum_wrapper(input.data(), output.data(), n);
            if (i > 0) cublas_dasum_time += time_instance;
            if (output.at(0) == n) {
                std::cout << "cublasDasum :: " << n << "-" << i << " \u2192 \u2713\n";  // good~!
            } else {
                std::cerr << "cublasDasum :: " << n << "-" << i << " \u2192 \u2717\n"; // bad~!
            }
        }

        cublas_dasum_time /= 9;

        oFile << n << "\t" 
              << my_dasum_time << "\t" 
              << cublas_dasum_time << "\n";
    }
    oFile.close();
}

void dnrm2_test() {

    std::cout << "dnrm2_tests" << sep << std::endl;

    std::ofstream oFile("dnrm2.dat");

    oFile << "n" << "\t" 
          << "my_dnrm2_time" << "\t"
          << "cublas_dnrm2_time" << "\n";

    float my_dnrm2_time, cublas_dnrm2_time, time_instance;

    for (unsigned int n = 2; n <= pow(2, 20); n = n * 2 + 1) {
        my_dnrm2_time = cublas_dnrm2_time = 0.0f;

        std::vector<double> input(n, 1.0f);
        std::vector<double> output(1, -1.0f);

        for (int i = 0; i < 10; i++) {
            time_instance = dnrm2(input.data(), output.data(), n);
            if (i > 0) my_dnrm2_time += time_instance;
            if (double_is_equal(output[0], sqrt(n))) {
                std::cout << "myDnrm2 :: " << n << "-" << i << " \u2192 \u2713\n";  // good~!
            } else {
                std::cerr << "myDnrm2 :: " << n << "-" << i << " \u2192 \u2717\n"; // bad~!
            }
        }
        my_dnrm2_time /= 9;

        for (int i = 0; i < 10; i++) {
            time_instance = cublas_dnrm2_wrapper(input.data(), output.data(), n);
            if (i > 0) cublas_dnrm2_time += time_instance;
            if (double_is_equal(output[0], sqrt(n))) {
                std::cout << "cublasDnrm2 :: " << n << "-" << i << " \u2192 \u2713\n";  // good~!
            } else {
                std::cerr << "cublasDnrm2 :: " << n << "-" << i << " \u2192 \u2717\n"; // bad~!
            }
        }

        cublas_dnrm2_time /= 9;

        oFile << n << "\t" 
              << my_dnrm2_time << "\t" 
              << cublas_dnrm2_time << "\n";
    }
    oFile.close();
}

void dgemm_test() {
    std::cout << "dgemm_tests" << sep << std::endl;


    std::ofstream oFile("dgemm.dat");

    oFile << "n" << "\t" 
          << "my_dgemm_time" << "\t"
          << "cublas_dgemm_time" << "\n";

    float my_dgemm_time, cublas_dgemm_time, time_instance;

    for (unsigned int n = 2; n <= 8192; n = n * 2 + 1) {
        const unsigned int size = n * n;
        std::vector<double> A(size, 0.0f);
        std::vector<double> B(size, 2.0f);
        std::vector<double> C(size, 0.0f);

        // make A an identity matrix
        for (unsigned int i = 0; i < n; i++) {
            A.at(n * i + i) = 1.0f;
        }

        // Init checksums
        std::vector<double> before_checksum(1, -1.0f);
        std::vector<double> after_checksum(1, -1.0f);

        for (int i = 0; i < 10; i++) {
            // Sum of B before Matmul 
            cublas_dasum_wrapper(B.data(), before_checksum.data(), size);
            // Perform A(identity) * B = C 
            // C should be equal to B after
            time_instance = dgemm(n, n, n, 1.0f, A.data(), B.data(), 0.0f, C.data());
            if (i > 0) my_dgemm_time += time_instance;
            // Sum of C after dgemm should be equal to B
            cublas_dasum_wrapper(C.data(), after_checksum.data(), size);
            if (before_checksum.at(0) != -1 && 
                after_checksum.at(0) != -1 &&
                double_is_equal(before_checksum.at(0), after_checksum.at(0))){
                std::cout << "myDgemm :: " << n << "-" << i << " \u2192 \u2713\n"; // good~!
            } else {
                std::cerr << "myDgemm :: " << n << "-" << i << " \u2192 \u2717 :: " << before_checksum.at(0) << " != " << after_checksum.at(0) << "\n"; // bad~!
            }
        }
        my_dgemm_time /= 9;

        for (int i = 0; i < 10; i++) {
            // Sum of B before Matmul 
            cublas_dasum_wrapper(B.data(), before_checksum.data(), size);
            // Perform A(identity) * B = C 
            // C should be equal to B after
            time_instance = cublas_dgemm_wrapper(n, n, n, 1.0f, A.data(), B.data(), 0.0f, C.data());
            if (i > 0) cublas_dgemm_time += time_instance;
            // Sum of C after dgemm should be equal to B
            cublas_dasum_wrapper(C.data(), after_checksum.data(), size);
            if (before_checksum.at(0) != -1 && 
                after_checksum.at(0) != -1 &&
                double_is_equal(before_checksum.at(0), after_checksum.at(0))){
                std::cout << "cublasDgemm :: " << n << "-" << i << " \u2192 \u2713\n"; // good~!
            } else {
                std::cerr << "cublasDgemm :: " << n << "-" << i << " \u2192 \u2717 :: " << before_checksum.at(0) << " != " << after_checksum.at(0) << "\n"; // bad~!
            }
        }
        cublas_dgemm_time /= 9;
        
        oFile << n << "\t" 
              << my_dgemm_time << "\t" 
              << cublas_dgemm_time << "\n";
    }
    oFile.close();
}

void daxpy_test() {
    std::cout << "daxpy_tests" << sep << std::endl;

    std::ofstream oFile("daxpy.dat");

    oFile << "n" << "\t" 
          << "my_daxpy_time" << "\t"
          << "cublas_daxpy_time" << "\n";

    float my_daxpy_time, cublas_daxpy_time, time_instance;

    // a * xi + yi = 1
    // 2 * 1 - 1  = 2
    double alpha = 2.0f;

    // Init checksums
    std::vector<double> checksum(1, -1.0f);

    for (unsigned int n = 2; n <= pow(2, 20); n = n * 2) {
        my_daxpy_time = cublas_daxpy_time = 0.0f;

        std::vector<double> x(n, 1.0f);
        std::vector<double> y(n, -1.0f);

        for (int i = 0; i < 10; i++) {
            time_instance = daxpy(n, alpha, x.data(), y.data());
            if (i > 0) my_daxpy_time += time_instance;
            // Sum of y (all 1s) after should be equal to n
            cublas_dasum_wrapper(y.data(), checksum.data(), n);
            if (checksum.at(0) == n) {
                std::cout << "mydaxpy :: " << n << "-" << i << " \u2192 \u2713\n";  // good~!
            } else {
                std::cerr << "mydaxpy :: " << n << "-" << i << "::  \u2192 \u2717 :: " << checksum.at(0) << " != " << n << "\n"; // bad~!
            }
            std::fill(y.begin(), y.end(), -1.0f);
        }
        my_daxpy_time /= 9;

        for (int i = 0; i < 10; i++) {
            time_instance = cublas_daxpy_wrapper(n, alpha, x.data(), y.data());
            if (i > 0) cublas_daxpy_time += time_instance;
            // Sum of y (all 1s) after should be equal to n
            cublas_dasum_wrapper(y.data(), checksum.data(), n);
            if (checksum.at(0) == n) {
                std::cout << "cublasdaxpy :: " << n << "-" << i << " \u2192 \u2713\n";  // good~!
            } else {
                std::cerr << "cublasdaxpy :: " << n << "-" << i << " \u2192 \u2717 :: " << checksum.at(0) << " != " << n << "\n"; // bad~!
            }
            std::fill(y.begin(), y.end(), -1.0f);
        }
        cublas_daxpy_time /= 9;

        oFile << n << "\t" 
              << my_daxpy_time << "\t" 
              << cublas_daxpy_time << "\n";
    }
    oFile.close();
}


void dcopy_test() {
    std::cout << "dcopy_tests" << sep << std::endl;

    std::ofstream oFile("dcopy.dat");

    oFile << "n" << "\t" 
          << "my_dcopy_time" << "\t"
          << "cublas_dcopy_time" << "\n";

    float my_dcopy_time, cublas_dcopy_time, time_instance;

    // Init checksum
    std::vector<double> checksum(1, -1.0f);

    for (unsigned int n = 2; n <= pow(2, 20); n = n * 2 + 1) {
        my_dcopy_time = cublas_dcopy_time = 0.0f;

        std::vector<double> x(n, 1.0f);
        std::vector<double> y(n, 0.0f);
        
        for (int i = 0; i < 10; i++) {
            time_instance = dcopy(n, x.data(), y.data());
        
            if (i > 0) my_dcopy_time += time_instance;
            cublas_dasum_wrapper(y.data(), checksum.data(), n);
            if (checksum.at(0) == n) {
                std::cout << "mydcopy :: " << n << "-" << i << " \u2192 \u2713\n";  // good~!
            } else {
                std::cerr << "mydcopy :: " << n << "-" << i << "::  \u2192 \u2717 :: " << checksum.at(0) << " != " << n << "\n"; // bad~!
            }
        }
        my_dcopy_time /= 9;

        for (int i = 0; i < 10; i++) {
            time_instance = cublas_dcopy_wrapper(n, x.data(), y.data());
            if (i > 0) cublas_dcopy_time += time_instance;
            // Sum of y after should be equal to what the sum was before if everything was copied
            cublas_dasum_wrapper(y.data(), checksum.data(), n);
            if (checksum.at(0) == n) {
                std::cout << "cublasdcopy :: " << n << "-" << i << " \u2192 \u2713\n";  // good~!
            } else {
                std::cerr << "cublasdcopy :: " << n << "-" << i << " \u2192 \u2717 :: " << checksum.at(0) << " != " << n << "\n"; // bad~!
            }
        }
        cublas_dcopy_time /= 9;

        oFile << n << "\t" 
              << my_dcopy_time << "\t" 
              << cublas_dcopy_time << "\n";
    }
    oFile.close();
}

int main() {
    dasum_test();
    dnrm2_test();
    dgemm_test();
    daxpy_test();
    dcopy_test();
    std::cout << "END OF MAIN" << std::endl;
}