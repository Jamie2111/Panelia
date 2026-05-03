FROM node:20-alpine

WORKDIR /app

COPY frontend/package.json /app/package.json
RUN npm install

COPY frontend /app

EXPOSE 3000

CMD ["npm", "run", "dev"]

