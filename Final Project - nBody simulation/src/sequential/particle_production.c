#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <time.h>

typedef struct {
    float mass;
    float x, y, z;
    float vx, vy, vz;
} Particle;

/* Function Declarations */
int convertStringToInt(char *str);
void randomizeParticles(Particle *particles, int n);

int main(int argc, char *argv[]) {
    int num_particles = 100000; // Default number of particles if no parameter is provided on the command line
    if (argc > 1) {
        // If a command-line parameter indicating the number of particles is provided, convert it to an integer
        num_particles = convertStringToInt(argv[1]);
    }

    // Allocate memory for particles
    Particle *particles = NULL;
    particles = (Particle *)malloc(num_particles * sizeof(Particle));

    // Start timer
    clock_t start = clock();

    // Randomize particles
    srand(0);
    randomizeParticles(particles, num_particles);

    // Write particles to file in binary mode
    FILE *file = fopen("particles.txt", "wb");
    if (file != NULL) {
        fwrite(particles, sizeof(Particle), num_particles, file);
        fclose(file);

        // Stop timer
        clock_t end = clock();
        double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

        printf("%d particles have been created with random values and written to file: sequential_output.bin in binary format.\n", num_particles);
        printf("Elapsed time for generating %d particles is: %f seconds.\n", num_particles, elapsed_time);
    }

    // Free allocated memory
    free(particles);
}

/* Convert string to integer */
int convertStringToInt(char *str) {
    char *endptr;
    long val;
    errno = 0; // To distinguish success/failure after the call

    val = strtol(str, &endptr, 10);

    /* Check for possible errors */
    if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) || (errno != 0 && val == 0)) {
        perror("strtol");
        exit(EXIT_FAILURE);
    }

    if (endptr == str) {
        fprintf(stderr, "No digits were found\n");
        exit(EXIT_FAILURE);
    }

    /* If we are here, strtol() successfully converted a number */
    return (int)val;
}

/* Initialize particle state with random values */
void randomizeParticles(Particle *particles, int n) {
    for (int i = 0; i < n; i++) {
        particles[i].mass = 2.0; // Arbitrarily chosen value for particle mass -> 2.0
        particles[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f; // Random number between -1 and 1
        particles[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        particles[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        particles[i].vx = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        particles[i].vy = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        particles[i].vz = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}
