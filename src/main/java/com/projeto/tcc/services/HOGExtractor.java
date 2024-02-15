package com.projeto.tcc.services;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.HOGDescriptor;
import org.springframework.stereotype.Service;

@Service
public class HOGExtractor {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    public float[] hogExtract(Mat frame) {
        //Já recebe a imagem carregada (Imgcodecs.imread)
        //Criação do objeto HOGDescriptor
        HOGDescriptor hog = new HOGDescriptor();
        //extracao das caracteristicas do frame
        MatOfFloat descriptor = new MatOfFloat();
        hog.compute(frame, descriptor);
        // Converte o descritor para vetor para facilitar a comparação e retorna
        return descriptor.toArray();
    }

    public double distanciaEuclidiana(float[] vector1, float[] vector2) {
        double distancia = 0;
        for (int i = 0; i < vector1.length; i++) {
            distancia += Math.pow(vector1[i] - vector2[i], 2);
        }
        return Math.sqrt(distancia);
    }
}
