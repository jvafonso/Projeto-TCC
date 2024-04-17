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

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

@Service
@Slf4j
public class HOGExtractor {

    private final HOGDescriptor hog;

    public HOGExtractor() {
        // Inicialização do HOGDescriptor
        hog = new HOGDescriptor();
    }

    public float[] hogExtract(Mat frame) {
        // Criação do objeto FloatPointer para armazenar o descritor
        FloatPointer descriptor = new FloatPointer();
        hog.compute(frame, descriptor);
        // Converte o descritor para vetor
        int descriptorSize = (int) (hog.getDescriptorSize());
        float[] descriptorArray = new float[descriptorSize];
        descriptor.get(descriptorArray);
        return descriptorArray;
    }

    public List<float[]> hogExtractBatch(List<Mat> frames) {
        List<float[]> descriptorsList = new ArrayList<>();
        for (Mat frame : frames) {
            descriptorsList.add(hogExtract(frame));
        }
        return descriptorsList;
    }


    public static double distanciaEuclidiana(float[] vector1, float[] vector2) {
        double distancia = 0;
        for (int i = 0; i < vector1.length; i++) {
            distancia += Math.pow(vector1[i] - vector2[i], 2);
        }
        double distanciaEuclidiana = Math.sqrt(distancia);

        // Normalização para o intervalo [0, 1] usando uma função sigmoide
        double similaridade = 1 / (1 + Math.exp(distanciaEuclidiana));

        if (similaridade > 1) {
            int value = (int)similaridade;
            similaridade = similaridade - value;
        }

        if (similaridade < 0.1){
            similaridade = similaridade * 10;
        }

        return similaridade;
    }


    public List<float[]> groupFramesHog(File descriptorFile, double similarityThreshold, double samplingPercentage) throws IOException {
        List<Group> groups = new ArrayList<>();
        List<float[]> selectedDescriptors = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(descriptorFile))) {
            List<String> descriptorLines = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    if (!descriptorLines.isEmpty()) {
                        float[] descriptor = parseDescriptor(descriptorLines);
                        descriptorLines.clear();
                        addToGroup(groups, descriptor, similarityThreshold);
                    }
                } else {
                    descriptorLines.add(line);
                }
            }
            if (!descriptorLines.isEmpty()) {
                float[] descriptor = parseDescriptor(descriptorLines);
                addToGroup(groups, descriptor, similarityThreshold);
            }
        }

        // Process groups and select a percentage of descriptors from each group
        for (Group group : groups) {
            group.descriptors.sort((d1, d2) -> Double.compare(distanciaEuclidiana(group.average, d2), distanciaEuclidiana(group.average, d1)));
            int elementsToSample = (int) (group.descriptors.size() * (samplingPercentage / 100.0));
            if (elementsToSample == 0 && !group.descriptors.isEmpty()) {
                elementsToSample = 1; // Garante que pelo menos um elemento seja selecionado
            }
            selectedDescriptors.addAll(group.descriptors.subList(0, elementsToSample));
            log.info(elementsToSample + " quadros adicionados à amostra do grupo.");
        }

        return selectedDescriptors;
    }

    public float[] parseDescriptor(List<String> descriptorLines) {
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

    private void addToGroup(List<Group> groups, float[] descriptor, double similarityThreshold) {
        for (Group group : groups) {
            if (group.isSimilar(descriptor, similarityThreshold)) {
                group.add(descriptor);
                return;
            }
        }
        groups.add(new Group(descriptor));
    }

    private static class Group {
        private List<float[]> descriptors = new ArrayList<>();
        private float[] average;

        Group(float[] descriptor) {
            add(descriptor);
        }

        void add(float[] descriptor) {
            descriptors.add(descriptor);
            updateAverage(descriptor);
        }

        boolean isSimilar(float[] descriptor, double similarityThreshold) {
            if (average == null) {
                return true;
            }
            double distance = distanciaEuclidiana(average, descriptor);
            log.info("Distancia: {}, similarityThreshold: {}", distance, similarityThreshold);
            return distance < similarityThreshold;
        }

        private void updateAverage(float[] newDescriptor) {
            if (average == null) {
                average = newDescriptor.clone();
            } else {
                for (int i = 0; i < average.length; i++) {
                    average[i] = (average[i] * (descriptors.size() - 1) + newDescriptor[i]) / descriptors.size();
                }
            }
        }

    }

    public List<float[]> sampleRandomDescriptors(File descriptorFile, double samplingPercentage) throws IOException {
        List<float[]> allDescriptors = new ArrayList<>();

        // Lendo e parseando todos os descritores do arquivo
        try (BufferedReader reader = new BufferedReader(new FileReader(descriptorFile))) {
            List<String> descriptorLines = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    if (!descriptorLines.isEmpty()) {
                        float[] descriptor = parseDescriptor(descriptorLines);
                        allDescriptors.add(descriptor);
                        descriptorLines.clear();
                    }
                } else {
                    descriptorLines.add(line);
                }
            }
            if (!descriptorLines.isEmpty()) {
                float[] descriptor = parseDescriptor(descriptorLines);
                allDescriptors.add(descriptor);
            }
        }

        // Calculando a quantidade de descritores a serem amostrados
        int totalDescriptors = allDescriptors.size();
        int sampleSize = (int) (totalDescriptors * samplingPercentage / 100.0);
        if (sampleSize == 0 && !allDescriptors.isEmpty()) {
            sampleSize = 1;  // Garante que pelo menos um elemento seja selecionado
        }

        // Embaralhando os descritores e selecionando a amostra
        Collections.shuffle(allDescriptors, new Random());
        List<float[]> sampledDescriptors = new ArrayList<>(allDescriptors.subList(0, Math.min(sampleSize, allDescriptors.size())));

        return sampledDescriptors;
    }


    public List<float[]> sampleFramesBySecondHOG(File descriptorFile, double samplingPercentage) throws IOException {
        List<float[]> allDescriptors = new ArrayList<>();

        // Lendo todos os descritores do arquivo
        try (BufferedReader reader = new BufferedReader(new FileReader(descriptorFile))) {
            List<String> descriptorLines = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    if (!descriptorLines.isEmpty()) {
                        float[] descriptor = parseDescriptor(descriptorLines);
                        allDescriptors.add(descriptor);
                        descriptorLines.clear();
                    }
                } else {
                    descriptorLines.add(line);
                }
            }
            if (!descriptorLines.isEmpty()) {
                float[] descriptor = parseDescriptor(descriptorLines);
                allDescriptors.add(descriptor);
            }
        }

        List<float[]> sampledDescriptors = new ArrayList<>();
        int framesPerSecond = 30; // Quantidade de quadros por segundo
        int sampleSize = (int) (framesPerSecond * (samplingPercentage / 100.0));

        for (int i = 0; i < allDescriptors.size(); i += framesPerSecond) {
            int end = Math.min(i + sampleSize, allDescriptors.size());
            for (int j = i; j < end; j++) {
                sampledDescriptors.add(allDescriptors.get(j));
            }
        }

        return sampledDescriptors;
    }


}
