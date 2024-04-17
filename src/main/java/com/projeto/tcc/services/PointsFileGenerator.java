package com.projeto.tcc.services;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

@Service
@Slf4j
public class PointsFileGenerator {

    @FunctionalInterface
    public interface DimensionalityFunction<T> {
        int getDimensionality(T descriptor);
    }

    @FunctionalInterface
    public interface DescriptorsToStringLinesFunction<T> {
        List<String> convertDescriptorsToStringLines(List<T> descriptors);
    }

    public void saveAsPointsFileHOG(List<float[]> descriptors, String filePath) throws IOException {
        saveAsPointsFile(descriptors, filePath, "HOG", this::getDimensionality, this::convertDescriptorsToStringLines);
    }

    public void saveAsPointsFileORB(List<Mat> descriptors, String filePath) throws IOException {
        int maxDimensionality = descriptors.stream()
                .mapToInt(Mat::cols)
                .max()
                .orElse(0);

        List<float[]> floatDescriptors = convertBinaryMatToFloatList(descriptors, maxDimensionality);
        saveAsPointsFile(floatDescriptors, filePath, "ORB", d -> maxDimensionality, this::convertDescriptorsToStringLines);
    }

    public void saveAsPointsFileCNN(List<INDArray> descriptors, String filePath) throws IOException {
        saveAsPointsFile(descriptors, filePath, "CNN", this::getCnnDimensionality, this::convertCnnDescriptorsToStringLines);
    }

    private <T> void saveAsPointsFile(List<T> descriptors, String filePath, String extractorName, DimensionalityFunction<T> dimensionalityFunction, DescriptorsToStringLinesFunction<T> converter) throws IOException {
        log.info("salvando Points File");
        log.info("Caminho do arquivo: " + filePath);
        log.info("Descriptores: {}", descriptors.isEmpty());
        if (descriptors.isEmpty()) {
            return;
        }

        int dimensionality = dimensionalityFunction.getDimensionality(descriptors.get(0));
        List<String> pointLines = converter.convertDescriptorsToStringLines(descriptors);
        log.info("pointLines: {}", pointLines.isEmpty());

        File baseDir = new File(filePath.substring(0, filePath.lastIndexOf(File.separator)));
        if (!baseDir.exists()) {
            baseDir.mkdirs();
        }

        int fileIndex = 1;
        String fileName = filePath + extractorName + fileIndex + ".data";
        while (new File(fileName).exists()) {
            fileIndex++;
            fileName = filePath + extractorName + fileIndex + ".data";
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            writer.write("DN\n"); // Dense representation without class
            writer.write(pointLines.size() + "\n"); // Number of points
            writer.write(dimensionality + "\n"); // Dimensionality of the points
            for (int i = 0; i < dimensionality; i++) {
                writer.write("D" + i + (i < dimensionality - 1 ? ";" : "\n"));
            }
            for (int i = 0; i < pointLines.size(); i++) {
                writer.write("P" + i + ";" + pointLines.get(i) + "\n");
            }
        } catch (IOException e) {
            log.error("Erro ao salvar o arquivo: " + e.getMessage(), e);
        }
        log.info("Arquivo salvo!!!");
    }

    private int getDimensionality(float[] descriptor) {
        return descriptor.length;
    }

    private List<String> convertDescriptorsToStringLines(List<float[]> descriptors) {
        List<String> pointLines = new ArrayList<>();
        for (float[] descriptor : descriptors) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < descriptor.length; i++) {
                sb.append(descriptor[i]).append(i < descriptor.length - 1 ? ";" : "");
            }
            pointLines.add(sb.toString());
        }
        return pointLines;
    }


    private int getCnnDimensionality(INDArray descriptor) {
        return (int) descriptor.length();
    }

    private List<String> convertCnnDescriptorsToStringLines(List<INDArray> descriptors) {
        log.info("convertCnnDescriptorsToStringLines");
        List<String> pointLines = new ArrayList<>();
        for (INDArray descriptor : descriptors) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < descriptor.length(); i++) {
                sb.append(descriptor.getDouble(i)).append(i < descriptor.length() - 1 ? ";" : "");
            }
            pointLines.add(sb.toString());
        }
        return pointLines;
    }

    public static List<float[]> convertBinaryDescriptorsToFloat(List<Mat> binaryDescriptors) {
        List<float[]> floatDescriptors = new ArrayList<>();
        for (Mat descriptor : binaryDescriptors) {
            // Calcula o número total de elementos no descritor (linhas * colunas)
            int totalElements = descriptor.rows() * descriptor.cols();
            float[] floatDescriptor = new float[totalElements];
            // Acessa os dados binários do descritor como um ponteiro de byte
            BytePointer bytePointer = new BytePointer(descriptor.data());
            // Converte cada byte em um valor de ponto flutuante e normaliza
            for (int i = 0; i < totalElements; i++) {
                floatDescriptor[i] = (bytePointer.get(i) & 0xFF) / 255.0f;
            }
            floatDescriptors.add(floatDescriptor);
        }
        return floatDescriptors;
    }

    public static List<float[]> convertBinaryMatToFloatList(List<Mat> binaryDescriptors, int requiredDimensionality) {
        List<float[]> floatDescriptors = new ArrayList<>();
        for (Mat descriptor : binaryDescriptors) {
            int totalElements = descriptor.rows() * descriptor.cols();
            float[] floatDescriptor = new float[requiredDimensionality];
            BytePointer bytePointer = new BytePointer(descriptor.data());
            int elementsToCopy = Math.min(totalElements, requiredDimensionality);
            for (int i = 0; i < elementsToCopy; i++) {
                // Mapeia 1 para 1.0 e 0 para 0.0
                floatDescriptor[i] = bytePointer.get(i) == 1 ? 1.0f : 0.0f;
            }
            // Preenche o restante com 0.0 se necessário
            for (int i = elementsToCopy; i < requiredDimensionality; i++) {
                floatDescriptor[i] = 0.0f;
            }
            floatDescriptors.add(floatDescriptor);
        }
        return floatDescriptors;
    }

    public void extractAndCopyFrames(String completeVideoPointsFilePath, String samplePointsFilePath, String framesDirectoryPath, String resultDirectoryPath) throws IOException {
        List<String> matchedLines = findMatchingLines(completeVideoPointsFilePath, samplePointsFilePath);
        copyCorrespondingFrames(framesDirectoryPath, resultDirectoryPath, matchedLines);
    }

    private List<String> findMatchingLines(String completeVideoPointsFilePath, String samplePointsFilePath) throws IOException {
        List<String> matchedLines = new ArrayList<>();
        List<String> sampleLines = readLinesStartingWithP(samplePointsFilePath);

        log.info("Iniciando a busca por linhas correspondentes.");
        try (BufferedReader reader = new BufferedReader(new FileReader(completeVideoPointsFilePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("P")) {
                    String lineContent = line.substring(line.indexOf(";") + 1); // Pega a parte da linha após o primeiro ";"
                    for (String sampleLine : sampleLines) {
                        String sampleLineContent = sampleLine.substring(sampleLine.indexOf(";") + 1);
                        if (lineContent.equals(sampleLineContent)) {
                            matchedLines.add(line.split(";")[0]);
                            log.info("Linha correspondente encontrada: " + line.split(";")[0]);
                            break;
                        }
                    }
                }
            }
        }
        log.info("Busca por linhas correspondentes concluída.");
        return matchedLines;
    }

    private List<String> readLinesStartingWithP(String filePath) throws IOException {
        List<String> lines = new ArrayList<>();
        log.info("Lendo linhas que começam com 'P' do arquivo: " + filePath);
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("P")) {
                    lines.add(line);
                }
            }
        }
        log.info("Leitura concluída. Total de linhas lidas: " + lines.size());
        return lines;
    }

    private void copyCorrespondingFrames(String framesDirectoryPath, String resultDirectoryPath, List<String> matchedLines) throws IOException {
        File resultDirectory = new File(resultDirectoryPath);
        if (!resultDirectory.exists()) {
            resultDirectory.mkdirs();
        }
        int contador = 0;
        log.info("Iniciando a cópia dos quadros correspondentes.");
        for (String line : matchedLines) {
            int frameNumber = Integer.parseInt(line.substring(1)); // Remove the "P" prefix
            String frameFileNamePng = "quadro (" + frameNumber + ").png";
            String frameFileNameJpg = "quadro (" + frameNumber + ").jpg";
            String newNameFile = "quadro (" + frameNumber + ") (P" + contador + ").jpg";
            File sourceFilePng = new File(framesDirectoryPath, frameFileNamePng);
            File sourceFileJpg = new File(framesDirectoryPath, frameFileNameJpg);
            File destinationFile = new File(resultDirectoryPath, newNameFile); // Assuming you want to save as JPG in the result directory

            if (sourceFilePng.exists()) {
                log.info("Copiando quadro PNG: " + frameFileNamePng);
                Files.copy(sourceFilePng.toPath(), destinationFile.toPath());
            } else if (sourceFileJpg.exists()) {
                log.info("Copiando quadro JPG: " + frameFileNameJpg);
                Files.copy(sourceFileJpg.toPath(), destinationFile.toPath());
            } else {
                log.warn("Quadro não encontrado: " + frameFileNamePng + " ou " + frameFileNameJpg);
            }
            contador++;
        }
        log.info("Cópia dos quadros concluída.");
    }

}
