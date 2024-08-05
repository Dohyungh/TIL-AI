import { Image, StyleSheet, Platform } from "react-native";

import { HelloWave } from "@/components/HelloWave";
import ParallaxScrollView from "@/components/ParallaxScrollView";
import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { TextInput } from "react-native";
import React from "react";
import axios from "axios";

export default function HomeScreen() {
  const onSubmitEdit = () => {
    // axios.post("", input)
    console.log(text);
  };
  const [text, onChangeText] = React.useState("");
  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: "#A1CEDC", dark: "#1D3D47" }}
      headerImage={
        <Image
          source={require("@/assets/images/cookie.jpg")}
          style={styles.reactLogo}
        />
      }
    >
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">On-device LLM Project: ODLM</ThemedText>
      </ThemedView>
      <ThemedView style={styles.stepContainer}>
        <ThemedText type="subtitle">Step 1: Front</ThemedText>
        <ThemedText>
          <ThemedText type="defaultSemiBold">
            React Native의 input 태그를 찾아보자.
          </ThemedText>{" "}
        </ThemedText>
        <TextInput
          style={styles.input}
          value={text}
          placeholder="AI에게 들려줄 말을 하고, STT로 바꿀 것입니다."
          onChangeText={onChangeText}
          onSubmitEditing={onSubmitEdit}
        />
      </ThemedView>
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  titleContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  stepContainer: {
    gap: 8,
    marginBottom: 8,
  },
  reactLogo: {
    height: 250,
    width: 450,
    bottom: 0,
    left: 0,
    position: "absolute",
  },
  input: {
    height: 40,
    margin: 12,
    borderWidth: 1,
    padding: 10,
  },
});
