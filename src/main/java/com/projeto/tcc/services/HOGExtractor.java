package com.projeto.tcc.services;




import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Size;
import org.opencv.core.MatOfFloat;
import org.bytedeco.opencv.global.opencv_objdetect;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.HOGDescriptor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@Service
@Slf4j
public class HOGExtractor {
    @Value("${opencv.native.library.path}")
    private String opencvNativeLibraryPath;

    public float[] hogExtract(Mat frame) {
        //Já recebe a imagem carregada (Imgcodecs.imread)
        //Criação do objeto HOGDescriptor
        log.info("Extração HOG");
        HOGDescriptor hog = new HOGDescriptor();
        FloatPointer descriptor = new FloatPointer();
        hog.compute(frame, descriptor);
        // Converte o descritor para vetor para facilitar a comparação e retorna
        int descriptorSize = (int) (hog.getDescriptorSize());
        float[] descriptorArray = new float[descriptorSize];
        descriptor.get(descriptorArray);
        //extracao das caracteristicas do frame
        log.info("Fim da Extração HOG");
        return descriptorArray;
    }

    public double distanciaEuclidiana(float[] vector1, float[] vector2) {
        double distancia = 0;
        for (int i = 0; i < vector1.length; i++) {
            distancia += Math.pow(vector1[i] - vector2[i], 2);
        }
        return Math.sqrt(distancia);
    }

    public static float[] calculateAverage(List<float[]> list) {
        // Verificar se a lista está vazia
        if (list == null || list.isEmpty()) {
            throw new IllegalArgumentException("A lista não pode ser vazia.");
        }

        // Inicializar o vetor de soma
        int length = list.get(0).length;
        float[] sum = new float[length];

        // Somar todos os vetores
        for (float[] array : list) {
            if (array.length != length) {
                throw new IllegalArgumentException("Todos os vetores devem ter o mesmo tamanho.");
            }
            for (int i = 0; i < length; i++) {
                sum[i] += array[i];
            }
        }

        // Calcular a média
        float[] average = new float[length];
        for (int i = 0; i < length; i++) {
            average[i] = sum[i] / list.size();
        }

        return average;
    }


    public List<float[]> groupFramesHog(File descriptorFile, double similarityThreshold) throws IOException {
        List<List<float[]>> groups = new ArrayList<>();
        List<float[]> selectedDescriptors = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(descriptorFile))) {
            List<String> descriptorLines = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    if (!descriptorLines.isEmpty()) {
                        float[] descriptor = parseDescriptor(descriptorLines);
                        descriptorLines.clear();
                        boolean addedToGroup = false;
                        for (List<float[]> group : groups) {
                            float[] groupDescriptor = calculateAverage(group);
                            double distance = distanciaEuclidiana(groupDescriptor, descriptor);
                            if (distance < similarityThreshold) {
                                group.add(descriptor);
                                addedToGroup = true;
                                break;
                            }
                        }
                        if (!addedToGroup) {
                            List<float[]> newGroup = new ArrayList<>();
                            newGroup.add(descriptor);
                            groups.add(newGroup);
                        }
                    }
                } else {
                    descriptorLines.add(line);
                }
            }
            // Process the last descriptor
            if (!descriptorLines.isEmpty()) {
                float[] descriptor = parseDescriptor(descriptorLines);
                boolean addedToGroup = false;
                for (List<float[]> group : groups) {
                    float[] groupDescriptor = calculateAverage(group);
                    double distance = distanciaEuclidiana(groupDescriptor, descriptor);
                    if (distance < similarityThreshold) {
                        group.add(descriptor);
                        addedToGroup = true;
                        break;
                    }
                }
                if (!addedToGroup) {
                    List<float[]> newGroup = new ArrayList<>();
                    newGroup.add(descriptor);
                    groups.add(newGroup);
                }
            }
        }

        // Create samples from groups
        for (List<float[]> group : groups) {
            float[] groupDescriptor = calculateAverage(group);
            float[] selectedDescriptor = null;
            double maxDistance = -1;
            for (float[] descriptor : group) {
                double distance = distanciaEuclidiana(groupDescriptor, descriptor);
                if (distance > maxDistance) {
                    maxDistance = distance;
                    selectedDescriptor = descriptor;
                }
            }
            if (selectedDescriptor != null) {
                selectedDescriptors.add(selectedDescriptor);
            }
        }

        return selectedDescriptors;
    }

    private float[] parseDescriptor(List<String> descriptorLines) {
        // Remove os colchetes de todas as linhas
        descriptorLines = descriptorLines.stream()
                .map(line -> line.replaceAll("\\[|\\]", ""))
                .collect(Collectors.toList());

        // Concatena as linhas e divide por vírgulas
        String allLines = String.join(" ", descriptorLines);
        String[] partsArray = allLines.split(",\\s*");
        float[] descriptor = new float[partsArray.length];
        for (int i = 0; i < partsArray.length; i++) {
            descriptor[i] = Float.parseFloat(partsArray[i]);
        }
        return descriptor;
    }



}
