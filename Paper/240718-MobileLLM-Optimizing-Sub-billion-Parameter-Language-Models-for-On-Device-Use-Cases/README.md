# MobileLLM: Optimizing Sub-billion Parameter language Models for On-Device Use Cases

[MobileLLM: Optimizing Sub-billion Parameter language Models for On-Device Use Cases](https://arxiv.org/pdf/2402.14905)

[github](https://github.com/facebookresearch/MobileLLM)

> SSAFY-특화 프로젝트 주제를 언어 모델로 정하고 나서,  
> Local로 언어 모델을 돌리는 것이 그래픽카드, 메모리 사양에 따라 불가능 할 때가 많다는 것을 들었던 것이 떠올랐다.  
> 찾아보니 실제로 40xx 시리즈 이상이 있어야 넉넉히 돌릴 수 있다는 것을 알게 됐고,  
> 원래 모바일 서비스로 기획할 수도 있다고 가능성을 열어두려던 차에 비상이 떨어졌다.  
> 급하게 논문을 찾았다.

## abstract

- mobile 배포를 목표로 했다.
- `MobileLLM`을 제안한다.
  - 깊고 가는 (lanky) 모델
  - embedding sharing
  - grouped-query attention mechanism
- 기존에 만연했던 "데이터" 와 "노드의 수" 가 모델의 성능을 좌우한다는 믿음을 깨고 모델의 "**구조**"에 중점을 두고 만들어 SOTA 모델을 얻었다.

## introduction

ChatGPT 같은 서비스는 모두 클라우드 컴퓨팅으로 작동한다.

그러나, 모바일 환경에서는 모든 사양이 downgrade 된다고 보면 된다.  
심지어, DRAM의 10% 이상 쓰면 안된다. 기계를 우리의 모델 혼자서 쓰지 않는다는 점을 명심해야 한다.
전력 또한 제한된다. 완충된 iphone에서 모델이 2시간 정도 돌아갔다는 언급이 있었다.

이 모든 제한사항들이 LLM 모델을 **축소**할 필요가 있음을 어필했고 본 연구가 이루어졌다. 1bilion: 10억개 보다 적은 parameter로 언어모델을 만들어보자!

- scaling law (Kaplan et al., 2020) 과는 모순되지만, 깊고 가는 모델이 더 성능이 좋았다.

- embedding sharing methods 와 grouped query attention을 이용해 weight utilization을 최대화시켰다.

- block-wise weight sharing 을 도입해 memory movement로 인한 병목을 해소했다.
  (`MobileLLM-LS`로 명명)

## training

32개의 A100 GPU를 썼다. (....)

## Building a Strong Baseline

### Feedforward Network Choice

기존에 흔한 FFN (FC - ReLU - FC) 보다 우수한 성능을 보이는 SwiGLU (Swish + GLU) 를 활성함수로 선택했다.

### 깊이냐 넓이냐

transformer를 쓰는 모델에서 넓이가 더 중요하다는 믿음이 **작은 모델** 에서는 통하지 않는다.

### embedding sharing

input 과 output의 embedding이 (vocab_size, embedding_dim)으로 공유된다.(??)

> we reuse the input embedding weights
> as the output fully connected layer weights

-> parameter 수를 줄이는 데 큰 도움을 줌

### transformer head size

transformer의 head 개수를 결정하는 주 요인은 더 많은 semantic 과 더 많은 비선형 head 조합 사이의 trade-off 이다.

그러나, 본 논문에서는 grouped query attention (원래는 key-value 캐시 사이즈를 줄이기 위해 고안된) 으로 key-value 헤드의 redundancy를 줄였다.
이는 일종의 weight sharing으로 볼 수 있다.

### Layer Sharing

깊이를 넓이보다 우선시하게 된 이후에, block들의 layer sharing 전략을 자연스럽게 떠올렸다.

1. 통과한 block을 즉시 재사용하거나
2. 일련의 block을 묶음으로 재사용하거나
3. 일련의 block을 뒤집어 재사용하거나

아래의 3가지 전략을 시험했다.

<p style="text-align: center">
<img src="./LayerSharing.png" style="width: 60%">
</p>

---

이후 실험 결과들과 performance 들이 이어진다.

생성된 dialogue 들이나 좀 보자.

```
Input:
Are there any particular physical benefits to mindful walking, such as improved posture or increased physical
fitness?
```

```
Output (MobileLLM-125M):
Yes, there are several physical benefits to mindful walking. Here are a few:
1. Improved Posture: Mindful walking can help improve your posture by promoting good alignment and reducing
stress on the joints. This can lead to better posture and reduced back pain.
2. Increased Flexibility: Mindful walking can also help increase flexibility in your body, which can reduce the
risk of injury and improve overall health.
3. Better Sleep: Mindful walking can help you fall asleep more easily and stay asleep longer, which can improve
your sleep quality.
4. Reduced Stress: Mindful walking can help reduce stress levels by promoting relaxation and reducing tension
in the body.
5. Enhanced Mental Health: Mindful walking has been shown to have positive effects on mental health, including
reducing symptoms of depression and anxiety.
Overall, mindful walking is an excellent way to improve your physical health and well-being. It’s a simple and
effective way to get moving and enjoy the benefits it offers. If you’re interested in trying mindful walking, I
would recommend starting with shorter walks and gradually increasing the duration over time. You may also
want to consider incorporating mindfulness into your daily routine to further enhance your experience.
In conclusion, mindful walking is a great way to improve your physical health and well-being. With its numerous
physical benefits, it’s easy to see why it’s becoming increasingly popular among people looking for a fun and
sustainable way to get moving. So go ahead and give it a try!
```

```
Output (MobileLLM-350M):
Yes, there are several physical benefits of mindful walking. Firstly, it can improve your posture by reducing the
amount of tension in your muscles and joints. Secondly, it can increase your physical fitness by improving your
cardiovascular health, strengthening your core muscles, and increasing your flexibility. Additionally, mindful
walking can help you to stay more focused on the present moment, which can lead to better sleep quality and
reduced stress levels. Finally, it can also help you to develop a greater sense of awareness and appreciation for
the world around you, which can enhance your overall well-being.
Overall, mindful walking is a great way to improve your physical health and well-being, while also promoting
relaxation and mindfulness. Whether you’re looking to reduce stress, improve your posture, or simply enjoy a
peaceful walk, mindful walking can be a rewarding and fulfilling experience.
If you’re interested in trying out mindful walking, there are many resources available that can guide you through
the process. Some popular options include guided meditations, apps that offer mindful walking exercises, and
classes that teach you how to practice mindful walking. With a little bit of effort and dedication, anyone can start
incorporating mindful walking into their daily routine and reap the many benefits it offers.
In conclusion, mindful walking is a simple yet powerful tool that can help you to improve your physical health,
reduce stress, and cultivate a deeper sense of awareness and appreciation for the world around you. So why not
give it a try and see what kind of benefits you can achieve? Your body and mind will thank you!
```

```
Output (llama-v2-7b-chat):
Mindful walking can have a range of physical benefits, including:
1. Improved posture: By focusing on the sensation of each step and the movement of the body, mindful walking
can help improve posture and reduce slouching or hunching.
2. Increased physical fitness: Mindful walking can help improve cardiovascular fitness, increase muscle strength
and endurance, and boost overall physical fitness.
3. Better balance and coordination: By paying attention to the body’s position in space and the movement of the
legs and feet, mindful walking can help improve balance and coordination.
4. Reduced stress and tension: Mindful walking can help reduce stress and tension in the body, which can lead to
improved physical relaxation and well-being.
5. Improved flexibility: By focusing on the movement of the body and the sensation of each step, mindful walking
can help improve flexibility and range of motion.
6. Boosted immune system: Studies have shown that mindfulness practices, including mindful walking, can help
boost the immune system and reduce inflammation.
7. Improved sleep: Mindful walking can help improve sleep quality and duration by reducing stress and promoting
relaxation. 8. Reduced pain: Mindful walking can help reduce pain by focusing on the sensation of each step and
the movement of the body, rather than the pain itself.
9. Improved mood: Mindful walking can help improve mood and reduce symptoms of depression and anxiety by
promoting relaxation and reducing stress.
It’s important to note that the physical benefits of mindful walking may vary depending on the individual and the
frequency and duration of the practice. However, with regular practice, mindful walking can have a range of
positive effects on physical health and well-being.
```

---

```
Input:
How do I wrap a present neatly?
```

```
Output (MobileLLM-125M):
1. Choose the right size gift: The size of the present you want to give will depend on your budget and personal
preference. If you are looking for a small gift, consider buying a small box or card with a single-use plastic bag.
2. Prepare the present: Make sure all the items in the present are clean and tidy. You can use a cleaning kit or a
tissue paper to remove any dirt or debris from the present.
3. Arrange the presents: Place the present in a place where it is easy to see and touch. Use a pencil to draw a line
around the present and arrange it in a way that makes sense to you.
4. Add decorations: Decorate the present with stickers, ribbons, or other decorative items. This will add some
personality to the present and make it more visually appealing.
5. Enjoy the present: Take time to enjoy the present and let yourself feel the warmth and comfort of the present.
Smile at yourself and take a moment to appreciate the present.
Remember, the most important thing is to have fun and enjoy the present! Happy gift-giving!
If you’re feeling adventurous, try making a surprise gift for someone special by wrapping the present in a cute
and colorful gift basket. Or, if you’re feeling adventurous, try making a gift for a friend who needs a little extra
help with their homework. Whatever you choose, remember to be creative and have fun with your present!
With these tips, you’ll be able to create a beautiful and thoughtful present that will last for years to come. Happy
gift-giving!
```

```
Output (MobileLLM-350M):
1. Start by wrapping the present in a decorative gift wrap or ribbon.
2. Use a small amount of glue or tape to secure the gift wrap or ribbon to the surface of the present.
3. Place the gift wrap or ribbon around the present, making sure it is centered and evenly distributed.
4. Use a small amount of glue or tape to secure the gift wrap or ribbon to the surface of the present.
5. Once the gift wrap or ribbon is securely attached, you can add any additional decorations or embellishments to
the present.
6. If you want to add a personal touch, you can write a message or a special message on the gift wrap or ribbon.
7. Finally, enjoy your beautifully wrapped present!
Remember, wrapping a gift is a fun and creative way to show your loved one how much you care. Have fun and
enjoy the process!

```

```
Output (llama-v2-7b-chat):
Wrapping a present neatly can be a bit of a challenge, but with a few simple tips and tricks, you can create a
beautifully wrapped gift that your recipient will love. Here are some tips to help you wrap a present neatly:
1. Use the right wrapping paper: Choose a wrapping paper that complements the gift and is the right size for the
present. You can also use a patterned paper or a solid color, depending on your preference.
2. Measure the gift: Before you start wrapping, measure the gift to make sure you have enough wrapping paper.
You can also use a piece of scrap paper to measure the gift and mark the edges with a pencil.
3. Use a flat surface: Make sure you have a flat surface to work on, such as a table or countertop. This will help
you keep the wrapping paper flat and prevent it from wrinkling.
4. Start with the edges: Begin by folding the edges of the wrapping paper over the gift, making sure they are even
and smooth. You can use tape to secure the edges if needed.
5. Roll the gift: Once you have the edges in place, roll the gift into the center of the wrapping paper. Make sure
the gift is securely in place before moving on to the next step.
6. Tuck in the edges: As you wrap the gift, tuck the edges of the wrapping paper under the gift, smoothing them
out as you go. Make sure the edges are even and smooth, and use tape to secure them if needed.
7. Add a ribbon or bow: Once you have the gift wrapped, add a ribbon or bow to the top. This will not only make
the gift look more festive, but it will also help to keep the wrapping paper in place.
8. Practice makes perfect: Wrapping a present neatly takes a bit of practice, so don’t be discouraged if it doesn’t
come out perfectly the first time. Keep practicing, and you’ll get the hang of it in no time!
By following these tips, you can create a beautifully wrapped present that your recipient will love. Happy
wrapping!
```

---

```
Input:
What are the best exercises for beginners?
```

```
Output (MobileLLM-125M):
1. Push-ups: This is a classic exercise that can be done anywhere, anytime. It’s easy to do and requires no
equipment.
2. Dumbbell Rows: These rows are great for building strength in your legs and core.
3. Squats: Squats are a great way to build strength in your legs and core.
4. Lunges: Lunges are a great way to improve balance and stability.
5. Plank: A plank is a great way to build strength in your back and core.
6. Jumping Jacks: Jumping jacks are a fun and effective way to build strength in your legs and core.
7. Burpees: Burpees are a great way to build strength in your legs and core.
8. Mountain Climbers: Mountain climbers are a great way to build strength in your legs and core.
9. Deadlifts: Deadlifts are a great way to build strength in your legs and core.
10. Tricep Dips: Tricep dips are a great way to build strength in your triceps.
Remember, it’s important to start with simple exercises and gradually increase the intensity as you get stronger.
Don’t push yourself too hard or you may find it difficult to maintain proper form. With consistent practice, you
will see improvements in your strength and overall fitness.
If you have any questions about exercises or healthy living, don’t hesitate to talk to a doctor or physical therapist.
They can help you develop a personalized plan that works best for you.
In conclusion, there are many exercises that can be done at home, but it’s important to start slowly and gradually
increase the intensity over time. Remember to listen to your body and take breaks when needed. And most
importantly, enjoy the process!
```

```
Output (MobileLLM-350M):
Beginners should start with exercises that are easy to do and don’t require a lot of strength or endurance. Here are
some exercises that are great for beginners:
1. Push-ups: This is a great exercise for beginners because it’s a simple exercise that can be done anywhere. Start
with a push-up on your knees and then lower yourself down to the ground.
2. Squats: Squats are a great exercise for building leg strength and endurance. Start with a squat on your knees
and then lower yourself down to the ground.
3. Lunges: Lunges are a great exercise for building leg strength and endurance. Start with a lunge on your front
foot and then lower yourself down to the ground.
4. Plank: Planks are a great exercise for building core strength and stability. Start with a plank on your forearms
and then lower yourself down to the ground.
5. Bicycle crunches: Bicycle crunches are a great exercise for building upper body strength and stability. Start
with a bicycle crunch on your upper back and then lower yourself down to the ground.
6. Plank with leg lifts: This exercise is great for building leg strength and stability. Start with a plank on your
forearms and then lift your legs off the ground and hold for a few seconds before lowering them back down.
7. Jumping jacks: Jumping jacks are a great exercise for building cardiovascular endurance and strength. Start
with a jumping jack on your feet and then jump up and spread your legs.
Remember to start with a weight that is comfortable for you and gradually increase the weight as you get stronger.
It’s also important to warm up before exercising and to stretch after your workout to prevent injury.
In conclusion, beginners should start with exercises that are easy to do and don’t require a lot of strength
or endurance. These exercises are great for building leg strength and endurance, and can be done anywhere.
Remember to start with a weight that is comfortable for you and gradually increase the weight as you get stronger.
Warming up and stretching after your workout are also important to prevent injury.
Following these exercises can help you build strength, endurance, and improve your overall fitness level.
Remember to always consult with a healthcare professional before starting any new exercise program.
```

```
Output (llama-v2-7b-chat):
Exercise is an essential part of a healthy lifestyle, but for beginners, it can be intimidating and overwhelming.
Here are some of the best exercises for beginners:
1. Bodyweight exercises: These exercises are great for beginners because they don’t require any equipment.
Examples include push-ups, squats, lunges, and planks.
2. Yoga: Yoga is a low-impact exercise that can help improve flexibility, balance, and strength. It’s also great for
reducing stress and improving overall well-being.
3. Brisk walking: Walking is a simple and accessible form of exercise that can be done anywhere. It’s a great
way to get your heart rate up and improve cardiovascular health.
4. Swimming: Swimming is a low-impact exercise that can help improve cardiovascular health and strengthen
muscles. It’s also easy on the joints, making it a great option for beginners.
5. Cycling: Cycling is another low-impact exercise that can be done indoors or outdoors. It’s a great way to
improve cardiovascular health and build leg strength.
6. Resistance band exercises: Resistance bands are lightweight and portable, making them easy to use at home or
on the go. They can help improve strength and flexibility in the upper body, lower body, and core.
7. Pilates: Pilates is a form of exercise that focuses on core strength, flexibility, and body control. It’s a great
option for beginners because it’s low-impact and can help improve posture and balance.
8. Dance-based workouts: Dance-based workouts, such as Zumba or Hip Hop Abs, are fun and energetic ways to
get in shape. They often involve aerobic exercise and strength training, making them a great option for beginners.
9. Jumping rope: Jumping rope is a high-intensity exercise that can help improve cardiovascular health and burn
calories. It’s also easy to do at home with a simple jump rope.
10. Kettlebell exercises: Kettlebells are small weights that can be used to perform a variety of exercises, including
swings, squats, and presses. They’re a great way to improve strength and cardiovascular health.
Remember to always consult with a doctor before starting any new exercise program, especially if you’re a
beginner. It’s also important to start slowly and gradually increase the intensity and duration of your workouts as
you become more comfortable.
```
